"""
RAG Pipeline for SalesBot Knowledge Base
Extracts text from the SalesBot FDD PDF and provides semantic search via TF-IDF.
"""

import os
import re
import logging
import numpy as np
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class PDFKnowledgeBase:
    """In-memory knowledge base built from a PDF document using TF-IDF for search."""

    def __init__(self, pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.vectorizer = None
        self.tfidf_matrix = None

        self._load()

    def _load(self):
        """Extract text from PDF, chunk it, and build TF-IDF index."""
        logger.info(f"Loading knowledge base from: {self.pdf_path}")

        if not os.path.exists(self.pdf_path):
            logger.error(f"PDF not found: {self.pdf_path}")
            self.chunks = ["Knowledge base PDF not found. Please ensure the document is available."]
            self._build_index()
            return

        text = self._extract_text()
        self.chunks = self._chunk_text(text)
        self._build_index()
        logger.info(f"Knowledge base loaded: {len(self.chunks)} chunks indexed")

    def _extract_text(self) -> str:
        """Extract all text from the PDF."""
        reader = PdfReader(self.pdf_path)
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n\n"
        return full_text

    def _chunk_text(self, text: str) -> list:
        """Split text into overlapping chunks, trying to break at paragraph boundaries."""
        # Clean up the text
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'Page \| \d+', '', text)
        text = re.sub(r'\.{3,}', '', text)  # Remove dot leaders from TOC
        text = re.sub(r'\s+', ' ', text).strip()

        # Split into paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 1 <= self.chunk_size:
                current_chunk += (" " + para) if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # If paragraph itself is larger than chunk_size, split it
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

        # Filter out very short chunks (likely noise)
        chunks = [c for c in chunks if len(c) > 30]

        return chunks if chunks else ["No content extracted from the document."]

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
            return ["No results found."]

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top_k indices sorted by similarity
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Minimum relevance threshold
                results.append({
                    "content": self.chunks[idx],
                    "relevance_score": round(float(similarities[idx]), 3)
                })

        if not results:
            return [{"content": "No relevant information found for your query.", "relevance_score": 0}]

        return results


# --- Singleton Instance ---
# Resolve PDF path relative to this file's directory
_pdf_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "SalesBot FDD v4.1 (1).pdf"
)

kb = PDFKnowledgeBase(_pdf_path)
