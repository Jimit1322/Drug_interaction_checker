import time
import hashlib
from google import genai
from google.genai import types
from backend.config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

# All free Gemini models — tries in order
MODELS_FALLBACK = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]

SYSTEM_PROMPT = """You are a clinical pharmacology assistant specializing 
in drug interactions. You answer questions about drug safety using the
provided source documents AND your medical knowledge.

Rules:
1. Check provided documents FIRST for interaction evidence
2. If documents show NO interaction between the drugs — this is meaningful.
   It likely means no significant interaction exists. Say so clearly.
3. If a drug interaction section exists for Drug A but doesn't mention Drug B,
   state that no interaction with Drug B is documented in FDA labeling
4. Always assign severity: MILD / MODERATE / SEVERE / NO KNOWN INTERACTION
5. Always cite sources
6. End with medical disclaimer

Response format:
⚠️ SEVERITY: [MILD/MODERATE/SEVERE/NO KNOWN INTERACTION]

[Clear explanation]

What to watch for:
- [symptom or risk]

Safer alternatives (if any):
- [alternative]

Sources:
📄 [Source name] — [section]

⚕️ This information is for educational purposes only.
   Always consult your pharmacist or doctor before changing medication."""


# ── In-memory response cache ──────────────────────────────────
_cache: dict[str, str] = {}

def _cache_key(question: str, docs: list[dict]) -> str:
    doc_ids = sorted([d.get("id", d["text"][:30]) for d in docs])
    raw     = question.lower().strip() + "|" + ",".join(doc_ids)
    return hashlib.md5(raw.encode()).hexdigest()


# ── Groq fallback (completely free, fast) ─────────────────────
def query_groq_fallback(prompt: str) -> str:
    """Last resort fallback using Groq free tier."""
    try:
        from groq import Groq
        import os
        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            return ""
        groq_client = Groq(api_key=groq_key)
        response    = groq_client.chat.completions.create(
            model    = "llama-3.1-70b-versatile",
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens  = 1024,
            temperature = 0.1,
        )
        print("  [Groq fallback used]")
        return response.choices[0].message.content
    except Exception as e:
        print(f"  [Groq failed] {e}")
        return ""


# ── Main function ─────────────────────────────────────────────
def query_gemini(user_question: str, retrieved_docs: list[dict]) -> str:
    # Check cache first — instant response for repeated queries
    cache_key = _cache_key(user_question, retrieved_docs)
    if cache_key in _cache:
        print("  [cache hit]")
        return _cache[cache_key]

    context = build_context(retrieved_docs)
    prompt  = f"""Here are the relevant medical documents:

{context}

---

User Question: {user_question}

Based on the documents above, provide a detailed answer about 
the drug interaction, following the format in your instructions."""

    last_error = ""

    # Try each Gemini model
    for model_name in MODELS_FALLBACK:
        for attempt in range(2):   # 2 attempts per model
            try:
                response = client.models.generate_content(
                    model    = model_name,
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
                        # automatic_function_calling = types.AutomaticFunctionCallingConfig(
                        #     disable = True
                        # ),
                    )
                )
                result = response.text
                _cache[cache_key] = result   # cache success
                if model_name != MODELS_FALLBACK[0]:
                    print(f"  [used: {model_name}]")
                return result

            except Exception as e:
                err        = str(e)
                last_error = err

                if any(code in err for code in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
                    wait = (attempt + 1) * 5   # 5s, 10s
                    print(f"  [{model_name}] busy, wait {wait}s (attempt {attempt+1}/2)")
                    time.sleep(wait)
                else:
                    # 404 or other — skip this model immediately
                    print(f"  [{model_name}] skipped: {err[:60]}")
                    break

    # Final fallback — Groq
    print("  All Gemini models busy — trying Groq...")
    groq_result = query_groq_fallback(prompt)
    if groq_result:
        _cache[cache_key] = groq_result
        return groq_result

    # Everything failed
    return (
        "⚠️ SEVERITY: UNKNOWN\n\n"
        "AI service is temporarily overloaded. Please try again in 1-2 minutes.\n\n"
        "⚕️ For urgent questions, consult your pharmacist directly."
    )


def build_context(retrieved_docs: list[dict]) -> str:
    if not retrieved_docs:
        return "No relevant documents found."
    parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.get("source", "Unknown")
        if source == "FDA DailyMed":
            header = (f"[Document {i}] FDA Drug Label\n"
                      f"Drug: {doc.get('drug_name', 'Unknown')}\n"
                      f"Section: {doc.get('section', 'Unknown')}")
        else:
            header = (f"[Document {i}] PubMed Research\n"
                      f"Title: {doc.get('title', 'Unknown')[:100]}\n"
                      f"Year: {doc.get('year', 'Unknown')} | PMID: {doc.get('pmid', 'Unknown')}")
        parts.append(f"{header}\n{doc['text']}")
    return "\n\n" + "─"*40 + "\n\n".join(parts)


def test_gemini_connection():
    try:
        for model in MODELS_FALLBACK:
            try:
                response = client.models.generate_content(
                    model    = model,
                    contents = "Say 'connected' only."
                )
                print(f"✅ {model}: {response.text.strip()}")
                return True
            except Exception as e:
                print(f"❌ {model}: {str(e)[:80]}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
    return False


if __name__ == "__main__":
    test_gemini_connection()