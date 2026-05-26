import asyncio
import base64
import json
import logging
import os
import re

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from gemini_live import GeminiLive
from twilio_handler import TwilioMediaBridge

# Load environment variables
load_dotenv()

# Configure logging - DEBUG for our modules, INFO for everything else
logging.basicConfig(level=logging.INFO)
logging.getLogger("gemini_live").setLevel(logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("MODEL", "gemini-3.1-flash-live-preview")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "+19785715824")

# ============ RAG & TOOL HANDLERS ============

from rag_pipeline import kb, DOCS_DIR

# In-memory store for qualified leads
qualified_leads = []

def handle_search_knowledge_base(**kwargs):
    """Search the SalesBot knowledge base PDF."""
    query = kwargs.get("query", "")
    results = kb.search(query, top_k=3)
    # Format results for the AI to read
    formatted = []
    for r in results:
        if isinstance(r, dict):
            formatted.append(r["content"])
        else:
            formatted.append(str(r))
    return {
        "query": query,
        "results": formatted,
        "num_results": len(formatted)
    }

def handle_qualify_lead(**kwargs):
    """Record lead qualification data."""
    lead = {
        "company_name": kwargs.get("company_name", "Unknown"),
        "contact_name": kwargs.get("contact_name", "Unknown"),
        "use_case": kwargs.get("use_case", "Not specified"),
        "team_size": kwargs.get("team_size", "Not specified"),
        "budget_range": kwargs.get("budget_range", "Not discussed"),
        "timeline": kwargs.get("timeline", "Not specified"),
        "status": "qualified"
    }
    qualified_leads.append(lead)
    logger.info(f"Lead qualified: {lead}")
    return {
        "success": True,
        "message": f"Lead for {lead['contact_name']} at {lead['company_name']} has been recorded.",
        "lead_id": f"LD-{len(qualified_leads):04d}",
        "details": lead
    }

def handle_schedule_demo(**kwargs):
    """Schedule a product demo."""
    from datetime import datetime
    demo = {
        "success": True,
        "demo_id": f"DEMO-{datetime.now().strftime('%Y%m%d')}-{len(qualified_leads)+1:03d}",
        "contact_name": kwargs.get("contact_name", ""),
        "email": kwargs.get("email", ""),
        "phone": kwargs.get("phone", ""),
        "preferred_date": kwargs.get("preferred_date", ""),
        "preferred_time": kwargs.get("preferred_time", ""),
        "duration": "30 minutes",
        "meeting_link": "https://meet.quantumbot.in/demo",
        "host": "Bikash Upadhaya (Product Lead)",
        "note": "A confirmation email will be sent shortly with the meeting details and calendar invite."
    }
    logger.info(f"Demo scheduled: {demo}")
    return demo


# Live transcript watchers (browser WebSockets watching phone calls)
live_watchers: set = set()

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for Gemini Live."""
    await websocket.accept()

    logger.info("WebSocket connection accepted")

    audio_input_queue = asyncio.Queue()
    video_input_queue = asyncio.Queue()
    text_input_queue = asyncio.Queue()

    client_disconnected = False

    async def audio_output_callback(data):
        if not client_disconnected:
            try:
                await websocket.send_bytes(data)
            except Exception:
                pass

    async def audio_interrupt_callback():
        pass

    gemini_client = GeminiLive(
        api_key=GEMINI_API_KEY,
        model=MODEL,
        input_sample_rate=16000,
        tool_mapping={
            "search_knowledge_base": handle_search_knowledge_base,
            "qualify_lead": handle_qualify_lead,
            "schedule_demo": handle_schedule_demo,
        }
    )

    session_task = None

    async def receive_from_client():
        nonlocal client_disconnected
        try:
            while True:
                message = await websocket.receive()

                if message.get("bytes"):
                    await audio_input_queue.put(message["bytes"])
                elif message.get("text"):
                    text = message["text"]
                    try:
                        payload = json.loads(text)
                        if isinstance(payload, dict) and payload.get("type") == "image":
                            logger.info(f"Received image chunk from client: {len(payload['data'])} base64 chars")
                            image_data = base64.b64decode(payload["data"])
                            await video_input_queue.put(image_data)
                            continue
                    except json.JSONDecodeError:
                        pass

                    await text_input_queue.put(text)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error receiving from client: {e}")
        finally:
            client_disconnected = True
            if session_task and not session_task.done():
                session_task.cancel()

    receive_task = asyncio.create_task(receive_from_client())

    MAX_RETRIES = 3
    RETRY_DELAYS = [2, 4, 8]

    async def run_session_with_retry():
        for attempt in range(MAX_RETRIES + 1):
            should_retry = False
            try:
                async for event in gemini_client.start_session(
                    audio_input_queue=audio_input_queue,
                    video_input_queue=video_input_queue,
                    text_input_queue=text_input_queue,
                    audio_output_callback=audio_output_callback,
                    audio_interrupt_callback=audio_interrupt_callback,
                ):
                    if event:
                        if event.get("type") == "error" and attempt < MAX_RETRIES:
                            error_msg = event.get("error", "")
                            if "exhausted" in error_msg or "quota" in error_msg.lower():
                                delay = RETRY_DELAYS[attempt]
                                logger.warning(f"Quota error, retrying in {delay}s (attempt {attempt+1}/{MAX_RETRIES})")
                                try:
                                    await websocket.send_json({"type": "status", "text": "Reconnecting..."})
                                except RuntimeError:
                                    return
                                await asyncio.sleep(delay)
                                should_retry = True
                                break
                        if event.get("type") == "go_away" and attempt < MAX_RETRIES:
                            logger.info(f"GoAway received, reconnecting (attempt {attempt+1}/{MAX_RETRIES})")
                            try:
                                await websocket.send_json({"type": "status", "text": "Reconnecting..."})
                            except RuntimeError:
                                return
                            await asyncio.sleep(1)
                            should_retry = True
                            break
                        try:
                            await websocket.send_json(event)
                        except RuntimeError:
                            return
                if not should_retry:
                    return
            except Exception as e:
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(f"Session error, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    raise

    try:
        session_task = asyncio.create_task(run_session_with_retry())
        await session_task
    except asyncio.CancelledError:
        logger.info("Gemini session cancelled due to client disconnect")
    except Exception as e:
        import traceback
        logger.error(f"Error in Gemini session: {type(e).__name__}: {e}\n{traceback.format_exc()}")
    finally:
        receive_task.cancel()
        try:
            await websocket.close()
        except:
            pass
        logger.info("connection closed")


# ============ TWILIO VOICE ENDPOINTS ============

@app.api_route("/twilio/voice", methods=["GET", "POST"])
async def twilio_voice(request: Request):
    """Twilio webhook: when someone calls your Twilio number, this answers."""
    host = request.headers.get("host", "localhost")
    protocol = "wss" if request.url.scheme == "https" or "onrender.com" in host or "globalvoxinc.ai" in host else "ws"
    ws_url = f"{protocol}://{host}/twilio/media-stream"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}">
            <Parameter name="caller" value="{{{{From}}}}" />
        </Stream>
    </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


@app.websocket("/twilio/media-stream")
async def twilio_media_stream(websocket: WebSocket):
    """WebSocket endpoint for Twilio Media Streams."""
    await websocket.accept()
    logger.info("Twilio Media Stream WebSocket accepted")

    gemini_client = GeminiLive(
        api_key=GEMINI_API_KEY,
        model=MODEL,
        input_sample_rate=16000,
        tool_mapping={
            "search_knowledge_base": handle_search_knowledge_base,
            "qualify_lead": handle_qualify_lead,
            "schedule_demo": handle_schedule_demo,
        }
    )

    async def broadcast_event(event):
        """Send transcript events to all live watchers."""
        dead = set()
        for watcher in live_watchers:
            try:
                await watcher.send_json(event)
            except Exception:
                dead.add(watcher)
        live_watchers.difference_update(dead)

    bridge = TwilioMediaBridge(
        websocket=websocket,
        gemini_client=gemini_client,
        text_trigger="A potential customer has connected. Please greet them and ask how you can help with SalesBot.",
        on_event=broadcast_event,
    )

    try:
        await bridge.run()
    except Exception as e:
        import traceback
        logger.error(f"Twilio bridge error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
    finally:
        try:
            await websocket.close()
        except:
            pass


@app.post("/call-me")
async def call_me(request: Request):
    """Make Twilio call a phone number and connect to the AI agent."""
    from twilio.rest import Client

    body = await request.json()
    to_number = body.get("phone")
    if not to_number:
        return {"error": "Missing 'phone' field. Send {\"phone\": \"+91XXXXXXXXXX\"}"}

    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        return {"error": "Twilio credentials not configured"}

    # Use PUBLIC_URL env var or Render URL — Twilio can't reach localhost
    public_url = os.getenv("PUBLIC_URL", "")
    if public_url:
        webhook_url = f"{public_url}/twilio/voice"
    else:
        host = request.headers.get("host", "localhost")
        protocol = "https" if "onrender.com" in host else request.url.scheme
        webhook_url = f"{protocol}://{host}/twilio/voice"

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=webhook_url,
        )
        logger.info(f"Outbound call initiated: {call.sid} to {to_number}")
        return {"success": True, "call_sid": call.sid, "to": to_number}
    except Exception as e:
        logger.error(f"Failed to initiate call: {e}")
        return {"error": str(e)}


# ============ LIVE TRANSCRIPT DASHBOARD ============

@app.get("/live")
async def live_dashboard():
    """Live transcript dashboard — watch phone calls in real-time."""
    return HTMLResponse(LIVE_DASHBOARD_HTML)


@app.websocket("/live/ws")
async def live_ws(websocket: WebSocket):
    """WebSocket for live transcript watchers."""
    await websocket.accept()
    live_watchers.add(websocket)
    logger.info(f"Live watcher connected ({len(live_watchers)} total)")
    try:
        while True:
            await websocket.receive_text()  # keep alive
    except:
        pass
    finally:
        live_watchers.discard(websocket)
        logger.info(f"Live watcher disconnected ({len(live_watchers)} total)")


LIVE_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Live Call Transcript</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0a0e17;
  --card: rgba(17,24,39,0.75);
  --border: rgba(255,255,255,0.08);
  --cyan: #00d4ff;
  --purple: #7c3aed;
  --green: #10b981;
  --red: #ef4444;
  --text: #f1f5f9;
  --muted: #64748b;
  --secondary: #94a3b8;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Inter', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
}
.top-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 24px;
  background: rgba(10,14,23,0.9);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 10;
}
.brand {
  font-weight: 700;
  font-size: 0.9rem;
  color: var(--cyan);
}
.status {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.75rem;
  font-weight: 600;
}
.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--muted);
}
.dot.live {
  background: var(--green);
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%,100% { opacity:1; box-shadow: 0 0 0 0 rgba(16,185,129,0.4); }
  50% { opacity:0.7; box-shadow: 0 0 0 4px rgba(16,185,129,0); }
}
.container {
  flex: 1;
  max-width: 700px;
  width: 100%;
  margin: 0 auto;
  padding: 20px;
  position: relative;
  z-index: 1;
}
.waiting {
  text-align: center;
  padding: 60px 20px;
  color: var(--muted);
}
.waiting h2 { font-size: 1.1rem; margin-bottom: 8px; color: var(--secondary); }
.waiting p { font-size: 0.8rem; }
#transcript {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.msg {
  padding: 10px 14px;
  border-radius: 12px;
  max-width: 85%;
  font-size: 0.875rem;
  line-height: 1.5;
  animation: fadeIn 0.2s ease-out;
}
@keyframes fadeIn {
  from { opacity:0; transform: translateY(8px); }
  to { opacity:1; transform: translateY(0); }
}
.msg .time {
  display: block;
  font-size: 0.6rem;
  opacity: 0.5;
  font-family: 'SF Mono', monospace;
  margin-top: 3px;
}
.msg.user {
  align-self: flex-end;
  background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(0,212,255,0.1));
  border: 1px solid rgba(0,212,255,0.15);
  border-bottom-right-radius: 4px;
}
.msg.gemini {
  align-self: flex-start;
  background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(124,58,237,0.1));
  border: 1px solid rgba(124,58,237,0.15);
  border-bottom-left-radius: 4px;
}
.msg.system {
  align-self: center;
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  color: var(--muted);
  font-size: 0.75rem;
  max-width: 100%;
  text-align: center;
}
.tool-card {
  align-self: center;
  background: rgba(16,185,129,0.08);
  border: 1px solid rgba(16,185,129,0.2);
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 0.75rem;
  color: var(--green);
  max-width: 100%;
  animation: fadeIn 0.2s ease-out;
}
.tool-card .tool-name { font-weight: 700; }
.tool-card pre {
  margin-top: 6px;
  color: var(--secondary);
  font-size: 0.7rem;
  white-space: pre-wrap;
  word-break: break-all;
}
</style>
</head>
<body>
<div class="top-bar">
  <span class="brand">Live Call Transcript</span>
  <div class="status">
    <span class="dot" id="statusDot"></span>
    <span id="statusText">Waiting for call...</span>
  </div>
</div>
<div class="container">
  <div class="waiting" id="waiting">
    <h2>No active call</h2>
    <p>Start a call using the "Call Me" button or dial +1 (978) 571-5824.<br>The transcript will appear here in real-time.</p>
  </div>
  <div id="transcript"></div>
</div>
<script>
const transcript = document.getElementById('transcript');
const waiting = document.getElementById('waiting');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
let currentUser = null;
let currentGemini = null;

const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
const ws = new WebSocket(protocol + '//' + location.host + '/live/ws');

ws.onopen = () => { statusText.textContent = 'Connected — waiting for call...'; };
ws.onclose = () => { statusText.textContent = 'Disconnected'; statusDot.className = 'dot'; };

ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);

  if (msg.type === 'call_start') {
    waiting.style.display = 'none';
    statusDot.className = 'dot live';
    statusText.textContent = 'Call in progress';
    addSystem('Call started');
    currentUser = null;
    currentGemini = null;
  }
  else if (msg.type === 'call_end') {
    statusDot.className = 'dot';
    statusText.textContent = 'Call ended';
    addSystem('Call ended');
    currentUser = null;
    currentGemini = null;
  }
  else if (msg.type === 'user') {
    if (currentUser) {
      currentUser.querySelector('.text').textContent += msg.text;
    } else {
      currentUser = addMsg('user', msg.text);
      currentGemini = null;
    }
  }
  else if (msg.type === 'gemini') {
    if (currentGemini) {
      currentGemini.querySelector('.text').textContent += msg.text;
    } else {
      currentGemini = addMsg('gemini', msg.text);
      currentUser = null;
    }
  }
  else if (msg.type === 'turn_complete') {
    currentUser = null;
    currentGemini = null;
  }
  else if (msg.type === 'tool_call') {
    addTool(msg.name, msg.result);
  }

  window.scrollTo(0, document.body.scrollHeight);
};

function addMsg(type, text) {
  const time = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
  const div = document.createElement('div');
  div.className = 'msg ' + type;
  div.innerHTML = '<span class="text"></span><span class="time">' + time + '</span>';
  div.querySelector('.text').textContent = text;
  transcript.appendChild(div);
  return div;
}

function addSystem(text) {
  const div = document.createElement('div');
  div.className = 'msg system';
  div.textContent = text;
  transcript.appendChild(div);
}

function addTool(name, result) {
  const div = document.createElement('div');
  div.className = 'tool-card';
  div.innerHTML = '<span class="tool-name">' + name + '</span><pre>' +
    JSON.stringify(result, null, 2).slice(0, 500) + '</pre>';
  transcript.appendChild(div);
}
</script>
</body>
</html>"""


# ============ ADMIN: RAG DOCUMENT MANAGEMENT ============

MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200 MB

@app.get("/admin")
async def admin_dashboard():
    """Admin dashboard for managing RAG documents."""
    return HTMLResponse(ADMIN_DASHBOARD_HTML)


@app.get("/admin/api/documents")
async def list_documents():
    """List all documents in the knowledge base."""
    docs = kb.get_documents()
    return {"documents": docs, "total_chunks": len(kb.chunks)}


@app.post("/admin/api/upload")
async def upload_document(request: Request):
    """Upload a document to the knowledge base (up to 200 MB)."""
    import shutil
    from fastapi import UploadFile

    form = await request.form()
    file = form.get("file")
    if not file:
        return {"error": "No file provided"}

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in {".pdf", ".txt", ".md"}:
        return {"error": f"Unsupported file type: {ext}. Only PDF, TXT, MD allowed."}

    # Safe filename
    safe_name = re.sub(r'[^\w\-.]', '_', filename)
    dest = os.path.join(DOCS_DIR, safe_name)

    # Check if file already exists
    if os.path.exists(dest):
        return {"error": f"File '{safe_name}' already exists. Delete it first to re-upload."}

    # Save file
    try:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_SIZE:
            return {"error": f"File too large ({len(contents) / 1024 / 1024:.1f} MB). Max is 200 MB."}

        with open(dest, "wb") as f:
            f.write(contents)

        # Reload knowledge base
        kb.reload()

        # Check if the uploaded file produced any chunks
        chunks_from_file = sum(1 for src in kb.chunk_sources if src == safe_name)

        if chunks_from_file == 0:
            # Remove the useless file
            os.remove(dest)
            kb.reload()
            return {
                "success": False,
                "warning": "No text could be extracted from this file. It may be a scanned/image-based PDF. Only text-based documents are supported.",
                "filename": safe_name,
                "deleted": True
            }

        return {
            "success": True,
            "filename": safe_name,
            "size_mb": round(len(contents) / (1024 * 1024), 2),
            "chunks_from_file": chunks_from_file,
            "total_chunks": len(kb.chunks)
        }
    except Exception as e:
        # Clean up on failure
        if os.path.exists(dest):
            os.remove(dest)
        return {"error": str(e)}


@app.delete("/admin/api/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document from the knowledge base."""
    filepath = os.path.join(DOCS_DIR, filename)

    if not os.path.exists(filepath):
        return {"error": f"File '{filename}' not found"}

    # Prevent path traversal
    if os.path.dirname(os.path.abspath(filepath)) != os.path.abspath(DOCS_DIR):
        return {"error": "Invalid filename"}

    try:
        os.remove(filepath)
        kb.reload()
        return {
            "success": True,
            "deleted": filename,
            "total_chunks": len(kb.chunks)
        }
    except Exception as e:
        return {"error": str(e)}


ADMIN_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SalesBot Admin - Knowledge Base</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0a0e17;
  --card: rgba(17,24,39,0.85);
  --border: rgba(255,255,255,0.08);
  --cyan: #00d4ff;
  --purple: #7c3aed;
  --green: #10b981;
  --red: #ef4444;
  --yellow: #f59e0b;
  --text: #f1f5f9;
  --muted: #64748b;
  --secondary: #94a3b8;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Inter', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
}
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
}

/* Top bar */
.top-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 28px;
  background: rgba(10,14,23,0.95);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 10;
}
.brand {
  font-weight: 700;
  font-size: 1rem;
  color: var(--cyan);
  display: flex;
  align-items: center;
  gap: 8px;
}
.brand small {
  font-weight: 400;
  opacity: 0.5;
  font-size: 0.8rem;
}
.back-link {
  color: var(--secondary);
  text-decoration: none;
  font-size: 0.8rem;
  padding: 6px 14px;
  border: 1px solid var(--border);
  border-radius: 8px;
  transition: all 0.15s;
}
.back-link:hover {
  color: var(--cyan);
  border-color: rgba(0,212,255,0.3);
}

/* Container */
.container {
  max-width: 800px;
  margin: 0 auto;
  padding: 32px 20px;
  position: relative;
  z-index: 1;
}

/* Stats bar */
.stats {
  display: flex;
  gap: 16px;
  margin-bottom: 28px;
}
.stat-card {
  flex: 1;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 20px;
  backdrop-filter: blur(16px);
}
.stat-label {
  font-size: 0.65rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin-bottom: 4px;
}
.stat-value {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--cyan);
}

/* Upload area */
.upload-area {
  background: var(--card);
  border: 2px dashed rgba(0,212,255,0.2);
  border-radius: 12px;
  padding: 40px 20px;
  text-align: center;
  margin-bottom: 28px;
  transition: all 0.2s;
  cursor: pointer;
  backdrop-filter: blur(16px);
}
.upload-area:hover, .upload-area.dragover {
  border-color: rgba(0,212,255,0.5);
  background: rgba(0,212,255,0.03);
}
.upload-area svg {
  color: var(--cyan);
  opacity: 0.6;
  margin-bottom: 12px;
}
.upload-area h3 {
  font-size: 0.95rem;
  margin-bottom: 6px;
}
.upload-area p {
  font-size: 0.75rem;
  color: var(--muted);
}
.upload-area input[type="file"] {
  display: none;
}
.upload-progress {
  margin-top: 16px;
  display: none;
}
.progress-bar {
  height: 4px;
  background: rgba(255,255,255,0.1);
  border-radius: 4px;
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--cyan), var(--purple));
  border-radius: 4px;
  transition: width 0.3s;
  width: 0%;
}
.upload-status {
  font-size: 0.75rem;
  margin-top: 8px;
  min-height: 1.2em;
}
.upload-status.success { color: var(--green); }
.upload-status.error { color: var(--red); }
.upload-status.loading { color: var(--cyan); }

/* Documents table */
.section-title {
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--cyan);
  margin-bottom: 12px;
}
.doc-list {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  backdrop-filter: blur(16px);
}
.doc-row {
  display: flex;
  align-items: center;
  padding: 14px 20px;
  gap: 16px;
  border-bottom: 1px solid var(--border);
  transition: background 0.15s;
}
.doc-row:last-child { border-bottom: none; }
.doc-row:hover { background: rgba(255,255,255,0.02); }

.doc-icon {
  width: 40px;
  height: 40px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.65rem;
  font-weight: 700;
  flex-shrink: 0;
}
.doc-icon.pdf { background: rgba(239,68,68,0.15); color: var(--red); }
.doc-icon.txt { background: rgba(16,185,129,0.15); color: var(--green); }
.doc-icon.md { background: rgba(124,58,237,0.15); color: var(--purple); }

.doc-info { flex: 1; min-width: 0; }
.doc-name {
  font-size: 0.85rem;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.doc-meta {
  font-size: 0.7rem;
  color: var(--muted);
  margin-top: 2px;
}

.doc-delete {
  background: rgba(239,68,68,0.1);
  border: 1px solid rgba(239,68,68,0.2);
  color: var(--red);
  padding: 6px 14px;
  border-radius: 8px;
  font-family: inherit;
  font-size: 0.75rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s;
  flex-shrink: 0;
}
.doc-delete:hover {
  background: rgba(239,68,68,0.2);
  border-color: rgba(239,68,68,0.4);
}
.doc-delete:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.empty-state {
  padding: 40px 20px;
  text-align: center;
  color: var(--muted);
  font-size: 0.85rem;
}

/* Toast */
.toast {
  position: fixed;
  bottom: 24px;
  right: 24px;
  padding: 12px 20px;
  border-radius: 10px;
  font-size: 0.8rem;
  font-weight: 600;
  z-index: 100;
  animation: slideIn 0.3s ease-out;
  display: none;
}
.toast.success { background: rgba(16,185,129,0.9); color: #fff; }
.toast.error { background: rgba(239,68,68,0.9); color: #fff; }
@keyframes slideIn {
  from { transform: translateY(20px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

@media (max-width: 600px) {
  .stats { flex-direction: column; gap: 8px; }
  .stat-card { padding: 12px 16px; }
  .stat-value { font-size: 1.2rem; }
  .doc-row { padding: 10px 14px; gap: 10px; }
  .doc-name { font-size: 0.8rem; }
  .doc-delete { padding: 5px 10px; font-size: 0.7rem; }
}
</style>
</head>
<body>
<div class="top-bar">
  <div class="brand">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
    Knowledge Base <small>Admin</small>
  </div>
  <a href="/" class="back-link">Back to SalesBot</a>
</div>

<div class="container">
  <!-- Stats -->
  <div class="stats">
    <div class="stat-card">
      <div class="stat-label">Documents</div>
      <div class="stat-value" id="docCount">-</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Indexed Chunks</div>
      <div class="stat-value" id="chunkCount">-</div>
    </div>
  </div>

  <!-- Upload -->
  <div class="upload-area" id="uploadArea">
    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/>
      <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
    <h3>Upload Document</h3>
    <p>Drag & drop or click to upload. PDF, TXT, MD up to 200 MB.</p>
    <input type="file" id="fileInput" accept=".pdf,.txt,.md" />
    <div class="upload-progress" id="uploadProgress">
      <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
      <div class="upload-status" id="uploadStatus"></div>
    </div>
  </div>

  <!-- Documents -->
  <div class="section-title">Documents</div>
  <div class="doc-list" id="docList">
    <div class="empty-state">Loading...</div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
const docList = document.getElementById('docList');
const docCount = document.getElementById('docCount');
const chunkCount = document.getElementById('chunkCount');
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const uploadStatus = document.getElementById('uploadStatus');
const toast = document.getElementById('toast');

// Load documents
async function loadDocs() {
  try {
    const res = await fetch('/admin/api/documents');
    const data = await res.json();
    docCount.textContent = data.documents.length;
    chunkCount.textContent = data.total_chunks;
    renderDocs(data.documents);
  } catch (e) {
    docList.innerHTML = '<div class="empty-state">Failed to load documents</div>';
  }
}

function renderDocs(docs) {
  if (!docs.length) {
    docList.innerHTML = '<div class="empty-state">No documents uploaded yet. Upload your first document above.</div>';
    return;
  }
  docList.innerHTML = docs.map(d => `
    <div class="doc-row">
      <div class="doc-icon ${d.type.toLowerCase()}">${d.type}</div>
      <div class="doc-info">
        <div class="doc-name">${escapeHtml(d.filename)}</div>
        <div class="doc-meta">${d.size_mb} MB</div>
      </div>
      <button class="doc-delete" onclick="deleteDoc('${escapeHtml(d.filename)}', this)">Delete</button>
    </div>
  `).join('');
}

// Upload
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadArea.classList.remove('dragover');
  if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) uploadFile(fileInput.files[0]);
});

async function uploadFile(file) {
  const maxSize = 200 * 1024 * 1024;
  if (file.size > maxSize) {
    showToast('File too large. Max 200 MB.', 'error');
    return;
  }

  uploadProgress.style.display = 'block';
  progressFill.style.width = '30%';
  uploadStatus.textContent = 'Uploading ' + file.name + '...';
  uploadStatus.className = 'upload-status loading';

  const formData = new FormData();
  formData.append('file', file);

  try {
    progressFill.style.width = '60%';
    const res = await fetch('/admin/api/upload', { method: 'POST', body: formData });
    const data = await res.json();
    progressFill.style.width = '100%';

    if (data.success) {
      uploadStatus.textContent = 'Uploaded! ' + data.chunks_from_file + ' chunks indexed.';
      uploadStatus.className = 'upload-status success';
      showToast(file.name + ' uploaded successfully (' + data.chunks_from_file + ' chunks)', 'success');
      loadDocs();
    } else if (data.warning) {
      uploadStatus.textContent = data.warning;
      uploadStatus.className = 'upload-status error';
      showToast(data.warning, 'error');
    } else {
      uploadStatus.textContent = data.error;
      uploadStatus.className = 'upload-status error';
      showToast(data.error, 'error');
    }
  } catch (e) {
    uploadStatus.textContent = 'Upload failed: ' + e.message;
    uploadStatus.className = 'upload-status error';
  }

  fileInput.value = '';
  setTimeout(() => { uploadProgress.style.display = 'none'; progressFill.style.width = '0%'; }, 3000);
}

// Delete
async function deleteDoc(filename, btn) {
  if (!confirm('Delete "' + filename + '"? The knowledge base will be re-indexed.')) return;
  btn.disabled = true;
  btn.textContent = 'Deleting...';

  try {
    const res = await fetch('/admin/api/documents/' + encodeURIComponent(filename), { method: 'DELETE' });
    const data = await res.json();
    if (data.success) {
      showToast(filename + ' deleted', 'success');
      loadDocs();
    } else {
      showToast(data.error, 'error');
      btn.disabled = false;
      btn.textContent = 'Delete';
    }
  } catch (e) {
    showToast('Delete failed', 'error');
    btn.disabled = false;
    btn.textContent = 'Delete';
  }
}

function showToast(msg, type) {
  toast.textContent = msg;
  toast.className = 'toast ' + type;
  toast.style.display = 'block';
  setTimeout(() => { toast.style.display = 'none'; }, 3000);
}

function escapeHtml(t) {
  const d = document.createElement('div');
  d.textContent = t;
  return d.innerHTML;
}

loadDocs();
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
