"""
RAG Pipeline for SalesBot Knowledge Base
Extracts text from PDF/TXT documents and provides semantic search via TF-IDF.
Supports multiple documents with add/remove/reload.
"""

import os
import re
import logging
import numpy as np
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Directory where uploaded documents are stored
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_docs")
os.makedirs(DOCS_DIR, exist_ok=True)


class KnowledgeBase:
    """In-memory knowledge base built from multiple documents using TF-IDF for search."""

    def __init__(self, docs_dir: str, chunk_size: int = 500):
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.chunks = []
        self.chunk_sources = []  # parallel list: which doc each chunk came from
        self.vectorizer = None
        self.tfidf_matrix = None

        self.reload()

    def reload(self):
        """Scan docs_dir, extract text from all docs, rebuild index."""
        self.chunks = []
        self.chunk_sources = []

        if not os.path.isdir(self.docs_dir):
            os.makedirs(self.docs_dir, exist_ok=True)

        files = self._list_files()
        if not files:
            self.chunks = ["No documents in knowledge base. Please upload documents via the admin dashboard."]
            self.chunk_sources = ["(none)"]
            self._build_index()
            logger.info("Knowledge base is empty — no documents found")
            return

        for filepath in files:
            filename = os.path.basename(filepath)
            try:
                text = self._extract_text(filepath)
                doc_chunks = self._chunk_text(text)
                self.chunks.extend(doc_chunks)
                self.chunk_sources.extend([filename] * len(doc_chunks))
                logger.info(f"Loaded {filename}: {len(doc_chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")

        if not self.chunks:
            self.chunks = ["Documents were found but no text could be extracted."]
            self.chunk_sources = ["(error)"]

        self._build_index()
        logger.info(f"Knowledge base loaded: {len(self.chunks)} chunks from {len(files)} documents")

    def _list_files(self) -> list:
        """List all supported files in docs_dir."""
        supported = {".pdf", ".txt", ".md"}
        files = []
        for f in os.listdir(self.docs_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext in supported:
                files.append(os.path.join(self.docs_dir, f))
        return sorted(files)

    def get_documents(self) -> list:
        """Return list of documents with metadata."""
        docs = []
        for f in os.listdir(self.docs_dir):
            filepath = os.path.join(self.docs_dir, f)
            if os.path.isfile(filepath):
                ext = os.path.splitext(f)[1].lower()
                size = os.path.getsize(filepath)
                docs.append({
                    "filename": f,
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 2),
                    "type": ext.lstrip(".").upper(),
                    "supported": ext in {".pdf", ".txt", ".md"},
                })
        return sorted(docs, key=lambda d: d["filename"])

    def _extract_text(self, filepath: str) -> str:
        """Extract text from a file based on its extension."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            return self._extract_pdf(filepath)
        elif ext in (".txt", ".md"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        return ""

    def _extract_pdf(self, filepath: str) -> str:
        """Extract all text from a PDF."""
        reader = PdfReader(filepath)
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n\n"
        return full_text

    def _chunk_text(self, text: str) -> list:
        """Split text into chunks."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'Page \| \d+', '', text)
        text = re.sub(r'\.{3,}', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 1 <= self.chunk_size:
                current_chunk += (" " + para) if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(para) > self.chunk_size:
                    words = para.split()
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk) + len(word) + 1 <= self.chunk_size:
                            current_chunk += (" " + word) if current_chunk else word
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = word
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        return [c for c in chunks if len(c) > 30]

    def _build_index(self):
        """Build TF-IDF index from chunks."""
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)

    def search(self, query: str, top_k: int = 3) -> list:
        """Search the knowledge base and return top_k most relevant chunks."""
        if not query or not self.chunks:
            return [{"content": "No results found.", "relevance_score": 0}]

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:
                results.append({
                    "content": self.chunks[idx],
                    "relevance_score": round(float(similarities[idx]), 3)
                })

        if not results:
            return [{"content": "No relevant information found for your query.", "relevance_score": 0}]

        return results


# --- Singleton Instance ---
kb = KnowledgeBase(DOCS_DIR)
