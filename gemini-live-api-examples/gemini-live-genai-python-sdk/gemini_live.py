import asyncio
import inspect
import logging
import traceback
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
from google import genai
from google.genai import types


def get_system_instruction():
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    day_after = today + timedelta(days=2)

    date_context = f"""## TODAY'S DATE
- Today is {today.strftime('%Y-%m-%d')} ({today.strftime('%A')}).
- Tomorrow = {tomorrow.strftime('%Y-%m-%d')} ({tomorrow.strftime('%A')}).
- Day after tomorrow = {day_after.strftime('%Y-%m-%d')} ({day_after.strftime('%A')}).
"""

    return date_context + SYSTEM_INSTRUCTION


SYSTEM_INSTRUCTION = """
## YOUR FIXED IDENTITY — DO NOT CHANGE
- Your name: Aria. NEVER use any other name — ONLY Aria.
- Company: QuantumBot (https://quantumbot.in/)
- Location: Ahmedabad, Gujarat, India
- You are an AI Sales Assistant for QuantumBot's product called "SalesBot" — an AI-powered Sales Automation Platform.

## YOUR ROLE
You handle inbound sales inquiries about SalesBot. Your goals are:
1. Answer product questions accurately using the knowledge base
2. Qualify potential leads
3. Schedule product demos with interested prospects

## ABSOLUTE FIRST STEP — NO EXCEPTIONS
When the call begins, greet the caller warmly and introduce yourself. Do NOT call any tool before speaking.

## OPENING LINE
"Hello! I'm Aria from QuantumBot. Thank you for your interest in SalesBot — our AI-powered sales automation platform. How can I help you today?"

## Language — HIGHEST PRIORITY RULE
- DEFAULT: English for the opening line.
- AUTO-DETECT FROM FIRST RESPONSE: As soon as the caller replies, detect their language and switch immediately:
  - If they speak Hindi/Hinglish → Switch to Hindi/Hinglish.
  - If they speak Gujarati → Switch to Gujarati.
  - If they speak English → Continue in English.
- After switching, STAY in that language for ALL subsequent responses.
- Do NOT mix languages after a switch.

## Your Voice & Personality
- Sound professional, friendly, and knowledgeable — like a real SaaS sales consultant.
- Be enthusiastic about the product but not pushy.
- Natural pace, natural pauses. Don't rush.

## Call Flow — Go step by step. Say ONE thing at a time, then WAIT for the caller to respond.

1. After greeting, ask what they'd like to know about SalesBot or what problem they're trying to solve.
   → WAIT for response.

2. When they ask about features, pricing, or capabilities — ALWAYS call the search_knowledge_base tool first with a relevant query. Use the results to answer accurately.
   → WAIT for response.

3. After answering 2-3 questions, naturally begin lead qualification:
   - Ask about their company and team size
   - Ask about their current sales/marketing process
   - Ask what channels they use (WhatsApp, Email, Phone)
   → Collect info step by step, WAIT between each question.

4. Once you have qualification info, call the qualify_lead tool to record it.

5. Offer to schedule a personalized demo:
   - "Would you like me to schedule a personalized demo with our product team? It's completely free and takes about 30 minutes."
   → WAIT for response.

6. If they agree, collect: name, email, phone, preferred date/time. Then call schedule_demo tool.

7. Close warmly: "Thank you for your interest in SalesBot! You'll receive a confirmation email shortly. Is there anything else I can help you with?"

## Tool Usage — MANDATORY
- Call search_knowledge_base BEFORE answering ANY question about the product, uploaded documents, or any topic the caller brings up. NEVER make up information — always check the knowledge base first.
- The knowledge base may contain various types of documents uploaded by the admin (product docs, invoices, reports, etc.). When the caller asks about ANY uploaded document or its contents, search the knowledge base and answer based on what you find.
- If the search results contain relevant information, use it to answer — even if it's not about SalesBot specifically.
- Call qualify_lead once you have enough info about the prospect.
- Call schedule_demo when the caller agrees to a demo and provides their details.

## KEY PRODUCT KNOWLEDGE (use search_knowledge_base for details)
SalesBot is a SaaS platform with these modules:
- Knowledge Base: Upload docs, text, URLs, Q&A to train the AI
- Agent Configuration: Set up bot personality, persona, prompts, rules
- Campaigns: WhatsApp & Email automation
- Templates: WhatsApp & Email templates
- Lead Management: Track, assign, and nurture leads
- Conversations: Unified inbox across channels
- Products: Catalog management linked to KB
- Integrations: WhatsApp Cloud API, Email (SMTP/IMAP)
- User Management: Role-based access, multi-org support
- Playground: Test agent behavior live

Plans: Trial, Growth, Professional (with credit-based billing)

## Rules
- NEVER make up product features, pricing, or capabilities. Only use what the knowledge base returns.
- NEVER use any name other than "Aria" for yourself.
- Keep responses to 2-3 sentences max. This is a phone call.
- Be helpful and patient. If someone is just exploring, that's fine — don't push for a demo.
- If the caller asks about ANYTHING — whether it's about SalesBot or about documents they've uploaded — ALWAYS call search_knowledge_base first. Use the search results to answer.
- If the knowledge base doesn't have the answer, say: "I don't have that specific detail right now, but I can have our product team follow up with you on that."
- Remember everything the caller says during the call.
- If caller is busy, offer to schedule a callback or send info via email.
"""

TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": "Search the SalesBot product knowledge base to find relevant information about features, pricing, capabilities, modules, and integrations. Call this BEFORE answering any product question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query — what the caller is asking about (e.g., 'pricing plans', 'WhatsApp integration', 'knowledge base features')"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "qualify_lead",
        "description": "Record lead qualification data after gathering information about the prospect.",
        "parameters": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "Prospect's company name"},
                "contact_name": {"type": "string", "description": "Prospect's name"},
                "use_case": {"type": "string", "description": "What they want to use SalesBot for (e.g., 'WhatsApp marketing', 'lead management', 'customer support')"},
                "team_size": {"type": "string", "description": "Number of people on their sales/marketing team"},
                "budget_range": {"type": "string", "description": "Their budget range if mentioned"},
                "timeline": {"type": "string", "description": "When they want to implement (e.g., 'immediately', 'next month', 'evaluating')"}
            },
            "required": ["company_name", "contact_name", "use_case"]
        }
    },
    {
        "name": "schedule_demo",
        "description": "Schedule a product demo when the caller agrees to see SalesBot in action.",
        "parameters": {
            "type": "object",
            "properties": {
                "contact_name": {"type": "string", "description": "Caller's name"},
                "email": {"type": "string", "description": "Caller's email address"},
                "phone": {"type": "string", "description": "Caller's phone number"},
                "preferred_date": {"type": "string", "description": "Preferred demo date"},
                "preferred_time": {"type": "string", "description": "Preferred demo time"}
            },
            "required": ["contact_name", "email", "preferred_date", "preferred_time"]
        }
    }
]

class GeminiLive:
    """
    Handles the interaction with the Gemini Live API.
    """
    def __init__(self, api_key, model, input_sample_rate, tools=None, tool_mapping=None):
        """
        Initializes the GeminiLive client.

        Args:
            api_key (str): The Gemini API Key.
            model (str): The model name to use.
            input_sample_rate (int): The sample rate for audio input.
            tools (list, optional): List of tools to enable. Defaults to None.
            tool_mapping (dict, optional): Mapping of tool names to functions. Defaults to None.
        """
        self.api_key = api_key
        self.model = model
        self.input_sample_rate = input_sample_rate
        self.client = genai.Client(api_key=api_key)
        self.tools = tools or [{"function_declarations": TOOLS}]
        self.tool_mapping = tool_mapping or {}

    async def start_session(self, audio_input_queue, video_input_queue, text_input_queue, audio_output_callback, audio_interrupt_callback=None):
        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Charon"
                    )
                )
            ),
            system_instruction=types.Content(parts=[types.Part(text=get_system_instruction())]),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=False,
                    start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                ),
                turn_coverage="TURN_INCLUDES_ONLY_ACTIVITY",
            ),
            tools=self.tools,
        )
        
        logger.info(f"Connecting to Gemini Live with model={self.model}")
        try:
          async with self.client.aio.live.connect(model=self.model, config=config) as session:
            logger.info("Gemini Live session opened successfully")
            
            async def send_audio():
                try:
                    while True:
                        chunk = await audio_input_queue.get()
                        await session.send_realtime_input(
                            audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={self.input_sample_rate}")
                        )
                except asyncio.CancelledError:
                    logger.debug("send_audio task cancelled")
                except Exception as e:
                    logger.error(f"send_audio error: {e}\n{traceback.format_exc()}")

            async def send_video():
                try:
                    while True:
                        chunk = await video_input_queue.get()
                        logger.info(f"Sending video frame to Gemini: {len(chunk)} bytes")
                        await session.send_realtime_input(
                            video=types.Blob(data=chunk, mime_type="image/jpeg")
                        )
                except asyncio.CancelledError:
                    logger.debug("send_video task cancelled")
                except Exception as e:
                    logger.error(f"send_video error: {e}\n{traceback.format_exc()}")

            async def send_text():
                try:
                    while True:
                        text = await text_input_queue.get()
                        logger.info(f"Sending text to Gemini: {text}")
                        await session.send_realtime_input(text=text)
                except asyncio.CancelledError:
                    logger.debug("send_text task cancelled")
                except Exception as e:
                    logger.error(f"send_text error: {e}\n{traceback.format_exc()}")

            event_queue = asyncio.Queue()

            async def receive_loop():
                try:
                    while True:
                        async for response in session.receive():
                            logger.debug(f"Received response from Gemini: {response}")
                            
                            # Log the raw response type for debugging
                            if response.go_away:
                                logger.warning(f"Received GoAway from Gemini: {response.go_away}")
                            if response.session_resumption_update:
                                logger.info(f"Session resumption update: {response.session_resumption_update}")
                            
                            server_content = response.server_content
                            tool_call = response.tool_call
                            
                            if server_content:
                                if server_content.model_turn:
                                    for part in server_content.model_turn.parts:
                                        if part.inline_data:
                                            if inspect.iscoroutinefunction(audio_output_callback):
                                                await audio_output_callback(part.inline_data.data)
                                            else:
                                                audio_output_callback(part.inline_data.data)
                                
                                if server_content.input_transcription and server_content.input_transcription.text:
                                    await event_queue.put({"type": "user", "text": server_content.input_transcription.text})
                                
                                if server_content.output_transcription and server_content.output_transcription.text:
                                    await event_queue.put({"type": "gemini", "text": server_content.output_transcription.text})
                                
                                if server_content.turn_complete:
                                    await event_queue.put({"type": "turn_complete"})
                                
                                if server_content.interrupted:
                                    if audio_interrupt_callback:
                                        if inspect.iscoroutinefunction(audio_interrupt_callback):
                                            await audio_interrupt_callback()
                                        else:
                                            audio_interrupt_callback()
                                    await event_queue.put({"type": "interrupted"})

                            if tool_call:
                                function_responses = []
                                for fc in tool_call.function_calls:
                                    func_name = fc.name
                                    args = fc.args or {}
                                    
                                    if func_name in self.tool_mapping:
                                        try:
                                            tool_func = self.tool_mapping[func_name]
                                            if inspect.iscoroutinefunction(tool_func):
                                                result = await tool_func(**args)
                                            else:
                                                loop = asyncio.get_running_loop()
                                                result = await loop.run_in_executor(None, lambda: tool_func(**args))
                                        except Exception as e:
                                            result = f"Error: {e}"
                                        
                                        function_responses.append(types.FunctionResponse(
                                            name=func_name,
                                            id=fc.id,
                                            response={"result": result}
                                        ))
                                        await event_queue.put({"type": "tool_call", "name": func_name, "args": args, "result": result})
                                
                                await session.send_tool_response(function_responses=function_responses)
                        
                        # session.receive() iterator ended (e.g. after turn_complete) — re-enter to keep listening
                        logger.debug("Gemini receive iterator completed, re-entering receive loop")

                except asyncio.CancelledError:
                    logger.debug("receive_loop task cancelled")
                except Exception as e:
                    logger.error(f"receive_loop error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                    await event_queue.put({"type": "error", "error": f"{type(e).__name__}: {e}"})
                finally:
                    logger.info("receive_loop exiting")
                    await event_queue.put(None)

            send_audio_task = asyncio.create_task(send_audio())
            send_video_task = asyncio.create_task(send_video())
            send_text_task = asyncio.create_task(send_text())
            receive_task = asyncio.create_task(receive_loop())

            try:
                while True:
                    event = await event_queue.get()
                    if event is None:
                        break
                    if isinstance(event, dict) and event.get("type") == "error":
                        # Just yield the error event, don't raise to keep the stream alive if possible or let caller handle
                        yield event
                        break 
                    yield event
            finally:
                logger.info("Cleaning up Gemini Live session tasks")
                send_audio_task.cancel()
                send_video_task.cancel()
                send_text_task.cancel()
                receive_task.cancel()
        except Exception as e:
            logger.error(f"Gemini Live session error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise
        finally:
            logger.info("Gemini Live session closed")
