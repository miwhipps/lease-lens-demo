import logging
from typing import List, Dict, Any
import re
import json
import pickle
import os
from collections import Counter
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeaseVectorStore:
    """Simple text-based vector store without ML dependencies"""

    def __init__(self, model_name="simple_tfidf"):
        self.model_name = model_name
        self.dimension = 100  # Simulated dimension

        # Document storage
        self.chunks = []
        self.texts = []
        self.metadata = []
        self.vocabulary = {}
        self.idf_scores = {}

        logger.info(f"✅ Simple vector store initialized: {model_name}")

    def chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 30) -> List[str]:
        """Split text into overlapping chunks for better search coverage"""
        logger.info(f"📝 Chunking text - size: {chunk_size}, overlap: {overlap}")

        # Clean up the input text first
        text = re.sub(r"\s+", " ", text).strip()

        # First try simple paragraph splitting for consistent results
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Convert paragraphs to sentences
        all_sentences = []
        for paragraph in paragraphs:
            # Split paragraph into sentences
            sentences = re.split(r"[.!?]+", paragraph)
            sentences = [s.strip() for s in sentences if s.strip()]

            all_sentences.extend(sentences)

        if not all_sentences:
            logger.warning("⚠️ No sentences found in text")
            return []

        chunks = []
        current_chunk = ""
        current_sentences = []

        for sentence in all_sentences:
            # Check word count (more accurate than character count)
            test_chunk = current_chunk + sentence + ". "
            word_count = len(test_chunk.split())

            if word_count <= chunk_size:
                # Add sentence to current chunk
                current_chunk = test_chunk
                current_sentences.append(sentence)
            else:
                # Current chunk is full, save it and start new one
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Start new chunk with overlap from previous chunk
                if overlap > 0 and current_sentences:
                    # Take last few sentences as overlap
                    overlap_sentences = current_sentences[-min(overlap // 10, len(current_sentences)) :]
                    current_chunk = ". ".join(overlap_sentences) + ". " + sentence + ". "
                    current_sentences = overlap_sentences + [sentence]
                else:
                    current_chunk = sentence + ". "
                    current_sentences = [sentence]

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Ensure we have meaningful chunks but don't over-merge
        final_chunks = []
        for chunk in chunks:
            word_count = len(chunk.split())
            if word_count >= 5:  # Lower minimum - keep more chunks
                final_chunks.append(chunk)
            elif final_chunks and len(final_chunks[-1].split()) < 150:  # Only merge if last chunk isn't already large
                final_chunks[-1] += " " + chunk
            else:  # Keep as separate chunk to maintain granularity
                final_chunks.append(chunk)

        logger.info(
            f"✅ Created {len(final_chunks)} chunks (avg {sum(len(c.split()) for c in final_chunks) // len(final_chunks) if final_chunks else 0} words each)"
        )
        return final_chunks

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple tokenization and cleaning
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

        # Remove common stop words
        stop_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
            "what",
            "where",
            "when",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "only",
            "own",
            "same",
            "than",
            "too",
            "very",
        }

        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords

    def add_document(self, text: str, doc_id: str, source_info: Dict[str, Any] = None):
        """Add document to the simple vector store"""
        logger.info(f"📄 Adding document: {doc_id}")

        if not text.strip():
            logger.warning("⚠️ Empty text provided, skipping")
            return

        # Chunk the text
        chunks = self.chunk_text(text)

        if not chunks:
            logger.warning("⚠️ No chunks created, skipping")
            return

        # Process each chunk
        for i, chunk in enumerate(chunks):
            keywords = self._extract_keywords(chunk)

            # Store chunk data
            chunk_data = {
                "text": chunk,
                "keywords": keywords,
                "keyword_counts": Counter(keywords),
                "doc_id": doc_id,
                "chunk_id": i,
                "source": source_info or {},
            }

            self.chunks.append(chunk_data)
            self.texts.append(chunk)
            self.metadata.append({"doc_id": doc_id, "chunk_id": i, "source": source_info or {}, "text": chunk})

            # Update vocabulary
            for keyword in keywords:
                if keyword not in self.vocabulary:
                    self.vocabulary[keyword] = len(self.vocabulary)

        # Calculate IDF scores
        self._calculate_idf()

        logger.info(f"✅ Added {len(chunks)} chunks to vector store")
        logger.info(f"📊 Total chunks: {len(self.texts)}")

    def _calculate_idf(self):
        """Calculate IDF scores for vocabulary"""
        total_docs = len(self.chunks)

        for word in self.vocabulary:
            # Count documents containing this word
            doc_count = sum(1 for chunk in self.chunks if word in chunk["keyword_counts"])

            # Calculate IDF score
            if doc_count > 0:
                self.idf_scores[word] = math.log(total_docs / doc_count)
            else:
                self.idf_scores[word] = 0

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using TF-IDF similarity"""
        if not self.chunks:
            logger.warning("⚠️ No documents in vector store")
            return []

        logger.info(f"🔍 Searching for: '{query}' (top {k} results)")

        # Extract query keywords
        query_keywords = self._extract_keywords(query)

        if not query_keywords:
            logger.warning("⚠️ No valid keywords in query")
            return []

        # Calculate similarities
        similarities = []

        for chunk in self.chunks:
            similarity = self._calculate_similarity(query_keywords, chunk)
            similarities.append(similarity)

        # Get top k results
        indexed_similarities = [(sim, idx) for idx, sim in enumerate(similarities)]
        indexed_similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for similarity, idx in indexed_similarities[:k]:
            # Lower threshold for including results - even low similarity might be useful
            if similarity > 0.01:  # Very low threshold to capture more results
                chunk = self.chunks[idx]
                results.append(
                    {
                        "text": chunk["text"],
                        "score": float(similarity),
                        "metadata": {"doc_id": chunk["doc_id"], "chunk_id": chunk["chunk_id"], "source": chunk["source"]},
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": chunk["doc_id"],
                    }
                )

        # If no good results found, apply fallback strategies
        if not results:
            logger.info("🔍 No similarity matches found, applying fallback strategies...")

            # Strategy 1: For financial queries, look for any numbers or currency symbols
            if any(word in query.lower() for word in ["rent", "payment", "cost", "monthly", "deposit", "fee"]):
                for idx, chunk in enumerate(self.chunks):
                    text_lower = chunk["text"].lower()
                    # Look for currency symbols, numbers, or financial keywords (including OCR errors)
                    if (
                        any(symbol in chunk["text"] for symbol in ["$", "£", "€", "¥"])
                        or any(
                            word in text_lower
                            for word in ["rent", "payment", "monthly", "cost", "deposit", "fee", "dollar", "pound"]
                        )
                        or
                        # Handle OCR errors in common financial terms
                        any(word in text_lower for word in ["r3nt", "r€nt", "m0nthly", "c0st", "d0llar"])
                        or
                        # Look for any numbers that might be amounts
                        any(char.isdigit() for char in chunk["text"])
                    ):
                        results.append(
                            {
                                "text": chunk["text"],
                                "score": 0.4,  # Moderate score for financial fallback
                                "metadata": {
                                    "doc_id": chunk["doc_id"],
                                    "chunk_id": chunk["chunk_id"],
                                    "source": chunk["source"],
                                },
                                "chunk_id": chunk["chunk_id"],
                                "doc_id": chunk["doc_id"],
                            }
                        )

            # Strategy 2: If still no results, return all chunks (last resort)
            if not results and len(self.chunks) > 0:
                logger.info("🔍 No financial matches found, returning all available chunks...")
                for idx, chunk in enumerate(self.chunks):
                    results.append(
                        {
                            "text": chunk["text"],
                            "score": 0.1,  # Low score for desperation fallback
                            "metadata": {"doc_id": chunk["doc_id"], "chunk_id": chunk["chunk_id"], "source": chunk["source"]},
                            "chunk_id": chunk["chunk_id"],
                            "doc_id": chunk["doc_id"],
                        }
                    )

        logger.info(f"✅ Found {len(results)} relevant chunks")
        return results

    def _calculate_similarity(self, query_keywords: List[str], chunk: Dict) -> float:
        """Calculate TF-IDF similarity between query and chunk"""
        if not query_keywords:
            return 0.0

        chunk_keywords = chunk["keyword_counts"]

        # Calculate similarity score
        score = 0.0
        query_length = len(query_keywords)

        for keyword in query_keywords:
            if keyword in chunk_keywords:
                # TF score (term frequency in chunk)
                tf = chunk_keywords[keyword] / len(chunk["keywords"]) if chunk["keywords"] else 0

                # IDF score
                idf = self.idf_scores.get(keyword, 0)

                # TF-IDF score
                tfidf = tf * idf
                score += tfidf

        # Normalize by query length
        normalized_score = score / query_length if query_length > 0 else 0

        # Add bonus for exact phrase matches
        query_text = " ".join(query_keywords)
        if query_text.lower() in chunk["text"].lower():
            normalized_score += 0.5

        # Add bonus for keyword density
        keyword_matches = sum(1 for kw in query_keywords if kw in chunk_keywords)
        density_bonus = keyword_matches / len(query_keywords) * 0.3

        # Enhanced topic-specific boosting
        chunk_text_lower = chunk["text"].lower()

        # Financial queries boost
        if any(word in query_keywords for word in ["rent", "payment", "monthly", "cost", "deposit", "fee"]):
            if any(symbol in chunk["text"] for symbol in ["$", "£"]) or any(
                word in chunk_text_lower for word in ["rent", "payment", "monthly", "cost", "deposit", "fee"]
            ):
                normalized_score += 0.4

        # Pet queries boost (handle both singular and plural)
        pet_query_words = ["pet", "pets", "animal", "animals", "dog", "dogs", "cat", "cats"]
        if any(word in query_keywords for word in pet_query_words):
            if any(
                word in chunk_text_lower
                for word in ["pet", "pets", "animal", "animals", "dog", "dogs", "cat", "cats", "breed", "vaccination"]
            ):
                normalized_score += 0.4

        # Utility queries boost
        utility_query_words = ["utility", "utilities", "electric", "electricity", "gas", "water"]
        if any(word in query_keywords for word in utility_query_words):
            if any(
                word in chunk_text_lower
                for word in ["utility", "utilities", "electric", "electricity", "gas", "water", "sewer", "internet"]
            ):
                normalized_score += 0.4

        # Parking queries boost
        parking_query_words = ["parking", "garage", "space", "spaces"]
        if any(word in query_keywords for word in parking_query_words):
            if any(word in chunk_text_lower for word in ["parking", "garage", "space", "spaces", "visitor", "assigned"]):
                normalized_score += 0.4

        # Maintenance queries boost
        if any(word in query_keywords for word in ["maintenance", "repair", "fix"]):
            if any(word in chunk_text_lower for word in ["maintenance", "repair", "fix", "emergency", "hvac", "plumbing"]):
                normalized_score += 0.4

        # Termination queries boost
        if any(word in query_keywords for word in ["terminate", "termination", "end", "notice"]):
            if any(word in chunk_text_lower for word in ["terminate", "termination", "notice", "penalty", "renewal"]):
                normalized_score += 0.4

        return min(normalized_score + density_bonus, 1.0)  # Cap at 1.0

    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        if not self.metadata:
            return {}

        docs = set(item["doc_id"] for item in self.metadata)

        stats = {
            "total_chunks": len(self.texts),
            "total_documents": len(docs),
            "average_chunk_length": sum(len(text) for text in self.texts) / len(self.texts) if self.texts else 0,
            "vocabulary_size": len(self.vocabulary),
            "model_name": self.model_name,
            "dimension": self.dimension,
        }

        return stats

    def save(self, path: str):
        """Save vector store to disk"""
        logger.info(f"💾 Saving vector store to: {path}")

        try:
            os.makedirs(path, exist_ok=True)

            # Save all data
            data = {
                "chunks": self.chunks,
                "texts": self.texts,
                "metadata": self.metadata,
                "vocabulary": self.vocabulary,
                "idf_scores": self.idf_scores,
                "model_name": self.model_name,
            }

            with open(os.path.join(path, "vectorstore.pkl"), "wb") as f:
                pickle.dump(data, f)

            # Save configuration
            config = {
                "model_name": self.model_name,
                "dimension": self.dimension,
                "total_chunks": len(self.texts),
                "vocabulary_size": len(self.vocabulary),
            }

            with open(os.path.join(path, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

            logger.info("✅ Vector store saved successfully")

        except Exception as e:
            logger.error(f"❌ Failed to save vector store: {e}")
            raise

    def load(self, path: str):
        """Load vector store from disk"""
        logger.info(f"📁 Loading vector store from: {path}")

        try:
            # Load data
            with open(os.path.join(path, "vectorstore.pkl"), "rb") as f:
                data = pickle.load(f)

            self.chunks = data["chunks"]
            self.texts = data["texts"]
            self.metadata = data["metadata"]
            self.vocabulary = data["vocabulary"]
            self.idf_scores = data["idf_scores"]
            self.model_name = data["model_name"]

            logger.info(f"✅ Loaded {len(self.texts)} chunks from disk")

        except Exception as e:
            logger.error(f"❌ Failed to load vector store: {e}")
            raise


# Test function
def test_vector_store():
    """Test the simple vector store"""
    logger.info("🧪 Testing simple vector store...")

    try:
        # Initialize
        vector_store = LeaseVectorStore()

        # Test document
        test_text = """
        RESIDENTIAL TENANCY AGREEMENT
        
        The monthly rent for this property is £1,500 per month.
        Security deposit required is £3,000, equivalent to two months rent.
        
        Pet Policy: Pets are allowed with written permission from landlord.
        Pet deposit of £500 per pet is required.
        Maximum of 2 pets allowed per unit.
        
        Utilities: Tenant is responsible for electricity and gas.
        Landlord pays for water, sewer, and rubbish collection.
        
        Parking: One assigned parking space is included in rent.
        Additional parking spaces available for £200 per month.
        
        Termination: Either party may terminate lease with 60 days written notice.
        Early termination penalty is two months rent.
        """

        # Add document
        vector_store.add_document(test_text, "test_lease", {"source": "test"})

        # Test searches
        test_queries = [
            "What is the monthly rent?",
            "Are pets allowed?",
            "What utilities do I pay?",
            "How much is parking?",
            "What is the termination notice?",
        ]

        for query in test_queries:
            results = vector_store.search(query, k=3)

            if results:
                logger.info(f"✅ Query: '{query}'")
                logger.info(f"   Top result: {results[0]['text'][:100]}...")
                logger.info(f"   Score: {results[0]['score']:.3f}")
            else:
                logger.warning(f"⚠️ No results for: '{query}'")

        # Show stats
        stats = vector_store.get_document_stats()
        logger.info(f"📊 Stats: {stats}")

        logger.info("✅ Simple vector store test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_vector_store()
