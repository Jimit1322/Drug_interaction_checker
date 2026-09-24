from google import genai
from google.genai import types
from backend.config import GEMINI_API_KEY, GEMINI_MODEL
import time

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """You are a clinical pharmacology assistant specializing 
in drug interactions. You answer questions about drug safety using only 
the provided source documents.

Rules you must follow:
1. Base your answer ONLY on the provided documents — never from general knowledge
2. Always assign a severity: MILD / MODERATE / SEVERE / UNKNOWN
3. Always cite your sources at the end using the format shown
4. If the documents don't contain enough information, say so clearly
5. Always end with the medical disclaimer
6. Never give a definitive "safe" or "unsafe" verdict — recommend consulting a doctor

Response format:
⚠️ SEVERITY: [MILD/MODERATE/SEVERE/UNKNOWN]

[Clear explanation of the interaction]

What to watch for:
- [symptom or risk 1]
- [symptom or risk 2]

Safer alternatives (if any):
- [alternative]

Sources:
📄 [Source name] — [section/detail]

⚕️ This information is for educational purposes only. 
   Always consult your pharmacist or doctor before changing medication."""


def query_gemini(user_question: str, retrieved_docs: list[dict]) -> str:
    context = build_context(retrieved_docs)
    prompt  = f"""Here are the relevant medical documents:

{context}

---

User Question: {user_question}

Based ONLY on the documents above, provide a detailed answer about 
the drug interaction, following the format in your instructions."""

    # Retry up to 3 times on 503
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model    = GEMINI_MODEL,
                contents = [
                    types.Content(
                        role  = "user",
                        parts = [types.Part(text=prompt)]
                    )
                ],
                config = types.GenerateContentConfig(
                    system_instruction = SYSTEM_PROMPT,
                    temperature        = 0.1,
                    max_output_tokens  = 2048,
                    automatic_function_calling = types.AutomaticFunctionCallingConfig(
                        disable = True
                    ),
                )
            )
            return response.text

        except Exception as e:
            err = str(e)
            if "503" in err and attempt < 2:
                print(f"  Gemini busy, retrying in 10s... (attempt {attempt+1}/3)")
                time.sleep(10)
                continue
            return f"Error generating response: {err}"
    
def build_context(retrieved_docs: list[dict]) -> str:
    if not retrieved_docs:
        return "No relevant documents found in the knowledge base."

    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.get("source", "Unknown")

        if source == "FDA DailyMed":
            header = (f"[Document {i}] FDA Drug Label\n"
                      f"Drug: {doc.get('drug_name', 'Unknown')}\n"
                      f"Section: {doc.get('section', 'Unknown')}")
        else:
            header = (f"[Document {i}] PubMed Research Paper\n"
                      f"Title: {doc.get('title', 'Unknown')[:100]}\n"
                      f"Year: {doc.get('year', 'Unknown')} | "
                      f"PMID: {doc.get('pmid', 'Unknown')}")

        context_parts.append(f"{header}\n{doc['text']}")

    return "\n\n" + "─" * 50 + "\n\n".join(context_parts)


def test_gemini_connection():
    try:
        response = client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = "Say 'Gemini connected successfully' only."
        )
        print("✅", response.text.strip())
        return True
    except Exception as e:
        print(f"❌ Gemini connection failed: {e}")
        return False


if __name__ == "__main__":
    test_gemini_connection()