import os
import numpy as np
from typing import Dict, List, Any
import logging
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeaseRAGAssistant:
    """RAG-based lease assistant with fixed Claude integration"""

    def __init__(self, vector_store, anthropic_api_key: str = None):
        self.vector_store = vector_store

        # Initialize Claude client with comprehensive error handling
        api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        logger.info(f"🔍 DEBUG: Initializing Claude client...")

        if not api_key:
            logger.warning("⚠️ No Claude API key provided - using fallback responses")
            logger.warning("⚠️ Set ANTHROPIC_API_KEY environment variable or pass api_key parameter")
            self.client = None
        else:
            logger.info(f"🔍 DEBUG: API key found, attempting client initialization...")
            try:
                self.client = self._initialize_claude_client(api_key)
                if self.client:
                    logger.info("✅ Claude client successfully initialized")
                    logger.info(f"✅ Client type: {type(self.client).__name__}")
                else:
                    logger.warning("⚠️ Claude client initialization returned None - using fallback responses")
                    logger.warning("⚠️ Check API key validity and network connectivity")
            except Exception as e:
                logger.error(f"❌ Claude client initialization exception: {type(e).__name__}: {e}")
                import traceback

                logger.error(f"❌ Full initialization traceback: {traceback.format_exc()}")
                self.client = None

        # System prompt for Claude
        self.system_prompt = """You are a helpful assistant that answers questions about lease documents.

IMPORTANT INSTRUCTIONS:
1. Use ONLY the information provided in the context below to answer questions
2. If the answer cannot be found in the context, say "I cannot find this information in the lease document"
3. Always provide citations by mentioning which part of the lease the information comes from
4. Be precise and factual - do not make assumptions or add information not in the context
5. If you find conflicting information, mention the discrepancy
6. For financial amounts, always include the exact figures mentioned
7. Keep responses concise but comprehensive
8. Use a professional but friendly tone

Context will be provided as numbered sources. Reference them as "Source 1", "Source 2", etc.
"""

    def _initialize_claude_client(self, api_key: str):
        """Initialize Claude client with comprehensive error handling and fallback patterns"""
        try:
            import anthropic
            import sys
            import os

            # Enhanced version debugging
            try:
                version = anthropic.__version__
                logger.info(f"🔍 DEBUG: Anthropic library version: {version}")
                logger.info(f"🔍 DEBUG: Python version: {sys.version}")
                logger.info(f"🔍 DEBUG: Anthropic module path: {anthropic.__file__}")
            except AttributeError as ve:
                logger.warning(f"⚠️ Could not determine anthropic library version: {ve}")

            # Log environment info for debugging
            logger.info(f"🔍 DEBUG: API key present: {'Yes' if api_key else 'No'}")
            logger.info(f"🔍 DEBUG: API key length: {len(api_key) if api_key else 0}")
            logger.info(f"🔍 DEBUG: Environment variables: ANTHROPIC_API_KEY={bool(os.getenv('ANTHROPIC_API_KEY'))}")

            # Pattern 1: Modern v0.8+ initialization (recommended)
            try:
                logger.info("🔄 Attempting Pattern 1: Modern v0.8+ initialization")
                client = anthropic.Anthropic(api_key=api_key)

                # Test the client with a simple call
                try:
                    logger.info("🔄 Testing client with simple API call...")
                    response = client.messages.create(
                        model="claude-3-haiku-20240307", max_tokens=10, messages=[{"role": "user", "content": "Hi"}]
                    )
                    logger.info("✅ Pattern 1 SUCCESS: Modern client initialized and tested")
                    return client
                except Exception as test_error:
                    logger.warning(f"⚠️ Pattern 1: Client created but test failed: {test_error}")
                    logger.warning(f"⚠️ Pattern 1: Test error type: {type(test_error).__name__}")
                    # Still return client as it might work for actual use
                    return client

            except TypeError as te:
                logger.error(f"❌ Pattern 1 FAILED: TypeError during initialization: {te}")
                logger.error(f"❌ Pattern 1: Error type: {type(te).__name__}")
            except Exception as e:
                logger.error(f"❌ Pattern 1 FAILED: {type(e).__name__}: {e}")

            # Pattern 2: Try with explicit timeout (some versions require this)
            try:
                logger.info("🔄 Attempting Pattern 2: With explicit timeout")
                client = anthropic.Anthropic(api_key=api_key, timeout=30.0)
                logger.info("✅ Pattern 2 SUCCESS: Client with timeout initialized")
                return client
            except Exception as e:
                logger.error(f"❌ Pattern 2 FAILED: {type(e).__name__}: {e}")

            # Pattern 3: Minimal parameters only
            try:
                logger.info("🔄 Attempting Pattern 3: Minimal parameters")
                client = anthropic.Anthropic(api_key)
                logger.info("✅ Pattern 3 SUCCESS: Minimal client initialized")
                return client
            except Exception as e:
                logger.error(f"❌ Pattern 3 FAILED: {type(e).__name__}: {e}")

            # Pattern 4: Try legacy Client class if available
            try:
                logger.info("🔄 Attempting Pattern 4: Legacy Client class")
                if hasattr(anthropic, "Client"):
                    client = anthropic.Client(api_key=api_key)
                    logger.info("✅ Pattern 4 SUCCESS: Legacy Client initialized")
                    return client
                else:
                    logger.info("ℹ️ Pattern 4: Legacy Client class not available")
            except Exception as e:
                logger.error(f"❌ Pattern 4 FAILED: {type(e).__name__}: {e}")

            # Pattern 5: Environment variable fallback
            try:
                logger.info("🔄 Attempting Pattern 5: Environment variable fallback")
                os.environ["ANTHROPIC_API_KEY"] = api_key
                client = anthropic.Anthropic()
                logger.info("✅ Pattern 5 SUCCESS: Environment variable client initialized")
                return client
            except Exception as e:
                logger.error(f"❌ Pattern 5 FAILED: {type(e).__name__}: {e}")

            logger.error("❌ ALL PATTERNS FAILED: Could not initialize Anthropic client")
            return None

        except ImportError as ie:
            logger.error(f"❌ Anthropic library not installed: {ie}")
            logger.error("❌ Install with: pip install anthropic==0.8.1")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error initializing Claude: {type(e).__name__}: {e}")
            logger.error(f"❌ Error details: {str(e)}")
            import traceback

            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return None

    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Query the lease assistant using RAG"""
        logger.info(f"❓ Processing query: '{question}'")

        # Retrieve relevant chunks
        results = self.vector_store.search(question, k=k)

        if not results:
            return {
                "answer": "I cannot find relevant information in the lease document to answer your question.",
                "sources": [],
                "confidence": 0.0,
                "method": "no_results",
            }

        # Prepare context from retrieved chunks
        context_parts = []
        for i, result in enumerate(results, 1):
            # Clean source text by removing page references and other noise
            clean_text = self._clean_source_text(result["text"])
            # Include chunk and document information for better source attribution
            doc_id = result.get("doc_id", "document")
            chunk_id = result.get("chunk_id", i - 1)
            context_parts.append(f"Source {i} (from {doc_id}, section {chunk_id}): {clean_text}")

        context = "\n\n".join(context_parts)

        # Generate answer
        if self.client:
            answer = self._generate_claude_answer(question, context)
            method = "claude"
        else:
            answer = self._generate_fallback_answer(question, results)
            method = "fallback"

        # Calculate confidence based on retrieval scores
        confidence = np.mean([r["score"] for r in results]) if results else 0.0

        response = {
            "answer": answer,
            "sources": results,
            "confidence": float(confidence),
            "method": method,
            "context_used": len(results),
        }

        logger.info(f"✅ Generated answer using {method} method (confidence: {confidence:.3f})")
        return response

    def _clean_source_text(self, text: str) -> str:
        """Clean source text by removing page references and other noise"""
        # Remove page references like "page 1 of 10", "Page 1/10", etc.
        import re

        # Remove common page reference patterns
        text = re.sub(r"\bpage\s+\d+\s+of\s+\d+\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bpage\s+\d+/\d+\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\b\d+\s+of\s+\d+\s+pages?\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bpg\.?\s+\d+\b", "", text, flags=re.IGNORECASE)

        # Remove excessive whitespace and normalize
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        # Clean up double punctuation left after page removal
        text = re.sub(r"[\.]{2,}", ".", text)  # Multiple periods to single
        text = re.sub(r"\.\s*\.", ".", text)  # Spaced double periods
        text = re.sub(r"\s*\.\s*\.", ".", text)  # Various double period patterns

        # Remove common OCR artifacts and formatting noise
        text = re.sub(r"^[-\s]+|[-\s]+$", "", text)  # Leading/trailing dashes
        text = re.sub(r"\s*\n\s*", "\n", text)  # Clean line breaks

        return text

    def _generate_claude_answer(self, question: str, context: str) -> str:
        """Generate answer using Claude API with better error handling"""
        try:
            # Prepare the user message
            user_message = f"""Based on the following lease document context, please answer the question accurately and cite your sources.

Context from lease document:
{context}

Question: {question}

Please provide a clear, accurate answer based only on the information in the context above. Include specific citations to the sources (e.g., "According to Source 1..." or "As stated in Source 2...").

Important: When citing sources, do not include any page references like "page X of Y" in your response. Focus on the actual content.

If the information needed to answer the question is not in the context, please say "I cannot find this information in the lease document."
"""

            # Modern API call with enhanced error handling
            try:
                logger.info(f"🔍 DEBUG: Making API call to Claude...")
                logger.info(f"🔍 DEBUG: Client type: {type(self.client).__name__}")
                logger.info(f"🔍 DEBUG: Model: claude-3-5-sonnet-20241022")

                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    temperature=0.1,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )

                logger.info(f"🔍 DEBUG: Response received, type: {type(response).__name__}")

                # Handle response format with detailed debugging
                if hasattr(response, "content") and len(response.content) > 0:
                    logger.info(f"🔍 DEBUG: Response has content, length: {len(response.content)}")
                    if hasattr(response.content[0], "text"):
                        answer = response.content[0].text.strip()
                        logger.info(f"🔍 DEBUG: Extracted text answer, length: {len(answer)}")
                    else:
                        # Handle different response formats
                        answer = str(response.content[0]).strip()
                        logger.info(f"🔍 DEBUG: Converted content to string, length: {len(answer)}")
                else:
                    answer = str(response).strip()
                    logger.info(f"🔍 DEBUG: Used full response as string, length: {len(answer)}")

                logger.info("✅ API call successful")
                return answer

            except Exception as api_error:
                logger.error(f"❌ API call failed: {type(api_error).__name__}: {api_error}")
                logger.error(f"❌ API error details: {str(api_error)}")

                # Try to provide more specific error information
                if hasattr(api_error, "status_code"):
                    logger.error(f"❌ HTTP status code: {api_error.status_code}")
                if hasattr(api_error, "response"):
                    logger.error(f"❌ Response content: {api_error.response}")

                import traceback

                logger.error(f"❌ API call traceback: {traceback.format_exc()}")

                return f"I found relevant information but couldn't generate a response due to an API error: {type(api_error).__name__}: {str(api_error)}"

        except Exception as e:
            logger.error(f"❌ Claude API call failed: {e}")
            return "I found relevant information but couldn't generate a proper response due to an API error. Please check the sources below for the information you're looking for."

    def _generate_fallback_answer(self, question: str, results: List[Dict]) -> str:
        """Generate enhanced fallback answer without Claude"""
        question_lower = question.lower()

        # Enhanced keyword-based answer generation with better formatting
        if any(word in question_lower for word in ["rent", "payment", "cost", "price", "monthly"]):
            for result in results:
                text = result["text"].lower()
                # Look for rent-related information more broadly
                if any(phrase in text for phrase in ["monthly rent", "rent:", "$", "£", "per month", "payment"]):
                    amounts = re.findall(r"[£$][\d,]+(?:\.\d{2})?", result["text"])
                    if amounts:
                        # Find the context around the amount
                        text_lines = result["text"].split("\n")
                        rent_context = ""
                        for line in text_lines:
                            if any(word in line.lower() for word in ["rent", "monthly", "$"]) and any(
                                amt in line for amt in amounts
                            ):
                                rent_context = line.strip()
                                break

                        if rent_context:
                            return f"**Monthly Rent Information:**\n{rent_context}\n\n*Source: \"{result['text'][:200]}...\"*"
                        else:
                            return f"**Monthly Rent Information:**\nThe rent amount is **{amounts[0]}** per month.\n\n*Source: \"{result['text'][:200]}...\"*"

        elif any(word in question_lower for word in ["pet", "animal", "dog", "cat"]):
            for result in results:
                if any(word in result["text"].lower() for word in ["pet", "animal", "dog", "cat"]):
                    return f"**Pet Policy:**\n{result['text'][:300]}...\n\n*Based on the lease document*"

        elif any(word in question_lower for word in ["break", "terminate", "end", "cancel", "notice"]):
            for result in results:
                if any(word in result["text"].lower() for word in ["break", "terminate", "notice"]):
                    return f"**Termination Clause:**\n{result['text'][:300]}...\n\n*From the lease agreement*"

        elif any(word in question_lower for word in ["deposit", "security"]):
            for result in results:
                if "deposit" in result["text"].lower():
                    amounts = re.findall(r"[£$][\d,]+(?:\.\d{2})?", result["text"])
                    if amounts:
                        return f"**Security Deposit:**\nThe security deposit is **{amounts[0]}**.\n\n*Source: \"{result['text'][:200]}...\"*"
                    return f"**Security Deposit Information:**\n{result['text'][:300]}..."

        elif any(word in question_lower for word in ["parking", "space"]):
            for result in results:
                if "parking" in result["text"].lower():
                    return f"**Parking Information:**\n{result['text'][:300]}...\n\n*From the lease agreement*"

        elif any(word in question_lower for word in ["utility", "utilities", "electric", "water"]):
            for result in results:
                if any(word in result["text"].lower() for word in ["utility", "utilities", "electric", "water", "gas"]):
                    return f"**Utilities Information:**\n{result['text'][:300]}...\n\n*Based on the lease terms*"

        elif any(word in question_lower for word in ["maintenance", "repair", "fix"]):
            for result in results:
                if any(word in result["text"].lower() for word in ["maintenance", "repair", "responsible"]):
                    return f"**Maintenance Responsibilities:**\n{result['text'][:300]}...\n\n*From the lease document*"

        elif any(word in question_lower for word in ["sublet", "subletting", "roommate"]):
            for result in results:
                if any(word in result["text"].lower() for word in ["sublet", "subletting", "permission"]):
                    return f"**Subletting Policy:**\n{result['text'][:300]}...\n\n*According to the lease agreement*"

        # Default response with top result
        if results:
            return (
                f"**Based on the lease document:**\n\n{results[0]['text'][:400]}...\n\n*Confidence: {results[0]['score']:.1%}*"
            )

        return "I cannot find specific information to answer your question in the lease document."

    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple questions at once"""
        logger.info(f"🔄 Processing batch of {len(questions)} questions")

        results = []
        for question in questions:
            result = self.query(question)
            results.append(result)

        return results

    def get_lease_summary(self) -> Dict[str, str]:
        """Generate a summary of key lease terms"""
        logger.info("📋 Generating lease summary")

        key_questions = [
            "What is the monthly rent amount?",
            "What is the security deposit?",
            "What is the lease term duration?",
            "What are the pet policies?",
            "What utilities are included?",
            "Are there any break clauses or early termination options?",
            "What are the parking arrangements?",
            "Who is responsible for maintenance?",
            "What are the late fees?",
            "Can the property be subletted?",
        ]

        summary = {}
        for question in key_questions:
            response = self.query(question, k=3)
            # Create proper category names from questions
            if "monthly rent" in question.lower():
                key = "Monthly Rent"
            elif "security deposit" in question.lower():
                key = "Security Deposit"
            elif "lease term" in question.lower():
                key = "Lease Term Duration"
            elif "pet policies" in question.lower():
                key = "Pet Policies"
            elif "utilities" in question.lower():
                key = "Utilities"
            elif "break clauses" in question.lower() or "termination" in question.lower():
                key = "Termination Options"
            elif "parking" in question.lower():
                key = "Parking Arrangements"
            elif "maintenance" in question.lower():
                key = "Maintenance Responsibilities"
            elif "late fees" in question.lower():
                key = "Late Fees"
            elif "sublet" in question.lower():
                key = "Subletting Policy"
            else:
                # Fallback: clean up the question as a key
                key = (
                    question.replace("What ", "")
                    .replace("?", "")
                    .replace("Are there any ", "")
                    .replace("Can the ", "")
                    .replace("Who is ", "")
                    .title()
                )

            summary[key] = response["answer"]

        return summary

    def analyze_lease_risks(self) -> List[Dict[str, Any]]:
        """Analyze potential risks or red flags in the lease"""
        logger.info("⚠️ Analyzing lease risks")

        risk_queries = [
            "Are there any unusual penalty clauses?",
            "What happens if rent is paid late?",
            "Are there any automatic renewal clauses?",
            "What are the subletting restrictions?",
            "Are there any additional fees mentioned?",
            "What are the termination penalties?",
            "Are there any damage liability clauses?",
            "What are the guest policy restrictions?",
        ]

        risks = []
        for query in risk_queries:
            response = self.query(query, k=2)
            if response["confidence"] > 0.4:
                risk_level = self._assess_risk_level(response["answer"])
                risks.append(
                    {
                        "category": query.replace("Are there any ", "").replace("What ", "").replace("?", ""),
                        "description": response["answer"],
                        "risk_level": risk_level,
                        "confidence": response["confidence"],
                        "sources": response["sources"][:2],
                    }
                )

        return risks

    def _assess_risk_level(self, answer: str) -> str:
        """Assess risk level based on answer content"""
        answer_lower = answer.lower()

        # High risk indicators
        high_risk_words = [
            "penalty",
            "forfeit",
            "immediate",
            "eviction",
            "legal action",
            "automatic renewal",
            "unlimited liability",
            "excessive",
        ]
        if any(word in answer_lower for word in high_risk_words):
            return "HIGH"

        # Medium risk indicators
        medium_risk_words = [
            "fee",
            "charge",
            "additional",
            "restriction",
            "prohibited",
            "written permission",
            "approval required",
            "not allowed",
        ]
        if any(word in answer_lower for word in medium_risk_words):
            return "MEDIUM"

        return "LOW"

    def extract_key_figures(self) -> Dict[str, Any]:
        """Extract key financial and date figures from the lease"""
        logger.info("💰 Extracting key figures")

        financial_queries = [
            "What is the monthly rent?",
            "What is the security deposit?",
            "What are the late fees?",
            "What are the pet fees?",
            "What are the parking costs?",
        ]

        date_queries = ["When does the lease start?", "When does the lease end?", "What is the lease duration?"]

        figures = {"financial": {}, "dates": {}, "summary": {}}

        # Extract financial information
        for query in financial_queries:
            response = self.query(query, k=2)
            key = query.replace("What is the ", "").replace("What are the ", "").replace("?", "")

            amounts = re.findall(r"[£$][\d,]+(?:\.\d{2})?", response["answer"])
            figures["financial"][key] = {
                "raw_answer": response["answer"],
                "amounts": amounts,
                "confidence": response["confidence"],
            }

        # Extract date information
        for query in date_queries:
            response = self.query(query, k=2)
            key = query.replace("When does the ", "").replace("What is the ", "").replace("?", "")

            figures["dates"][key] = {"raw_answer": response["answer"], "confidence": response["confidence"]}

        # Generate summary
        total_monthly_cost = self._calculate_total_monthly_cost(figures["financial"])
        figures["summary"] = {
            "estimated_monthly_cost": total_monthly_cost,
            "move_in_cost": self._calculate_move_in_cost(figures["financial"]),
        }

        return figures

    def _calculate_total_monthly_cost(self, financial_data: Dict) -> str:
        """Calculate estimated total monthly cost"""
        try:
            total = 0

            # Add rent
            rent_amounts = financial_data.get("monthly rent", {}).get("amounts", [])
            if rent_amounts:
                rent = float(rent_amounts[0].replace("$", "").replace("£", "").replace(",", ""))
                total += rent

            # Add parking if monthly
            parking_amounts = financial_data.get("parking costs", {}).get("amounts", [])
            parking_answer = financial_data.get("parking costs", {}).get("raw_answer", "")
            if parking_amounts and "month" in parking_answer.lower():
                parking = float(parking_amounts[0].replace("$", "").replace("£", "").replace(",", ""))
                total += parking

            # Add pet fees if monthly
            pet_amounts = financial_data.get("pet fees", {}).get("amounts", [])
            pet_answer = financial_data.get("pet fees", {}).get("raw_answer", "")
            if pet_amounts and "month" in pet_answer.lower():
                pet_fee = float(pet_amounts[0].replace("$", "").replace("£", "").replace(",", ""))
                total += pet_fee

            return f"${total:,.2f}/month" if total > 0 else "Unable to calculate"

        except Exception as e:
            logger.error(f"Error calculating monthly cost: {e}")
            return "Unable to calculate"

    def _calculate_move_in_cost(self, financial_data: Dict) -> str:
        """Calculate estimated move-in cost"""
        try:
            total = 0

            # Add first month rent
            rent_amounts = financial_data.get("monthly rent", {}).get("amounts", [])
            if rent_amounts:
                rent = float(rent_amounts[0].replace("$", "").replace("£", "").replace(",", ""))
                total += rent

            # Add security deposit
            deposit_amounts = financial_data.get("security deposit", {}).get("amounts", [])
            if deposit_amounts:
                deposit = float(deposit_amounts[0].replace("$", "").replace("£", "").replace(",", ""))
                total += deposit

            return f"${total:,.2f}" if total > 0 else "Unable to calculate"

        except Exception as e:
            logger.error(f"Error calculating move-in cost: {e}")
            return "Unable to calculate"


# Test function
def test_rag_assistant():
    """Test RAG assistant functionality"""
    logger.info("🧪 Testing RAG assistant...")

    # Mock vector store for testing
    class MockVectorStore:
        def search(self, query, k=5):  # pylint: disable=unused-argument
            return [
                {
                    "text": "Monthly rent is $2,000 per month, due on the 1st of each month. Late fees of $50 apply after the 5th day.",
                    "score": 0.95,
                    "metadata": {"doc_id": "test_lease", "chunk_id": 0},
                }
            ]

    # Test assistant
    mock_vector_store = MockVectorStore()
    assistant = LeaseRAGAssistant(mock_vector_store)

    # Test query
    response = assistant.query("What is the monthly rent?")

    if response and response["answer"]:
        logger.info("✅ RAG assistant test passed!")
        logger.info(f"   Answer: {response['answer'][:100]}...")
        logger.info(f"   Confidence: {response['confidence']:.3f}")
        logger.info(f"   Method: {response['method']}")
        return True
    else:
        logger.error("❌ RAG assistant test failed!")
        return False


if __name__ == "__main__":
    test_rag_assistant()
