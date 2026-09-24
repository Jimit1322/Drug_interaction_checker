# test_gemini.py
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
print(f"Key loaded: {key[:10]}..." if key else "❌ Key is None")

client = genai.Client(api_key=key)
response = client.models.generate_content(
    model    = "gemini-3.6-flash",
    contents = "Say hello"
)
print("✅", response.text)