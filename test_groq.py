
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

models = [
    os.getenv("GROQ_MODEL_1", "openai/gpt-oss-20b"),
    os.getenv("GROQ_MODEL_2", "qwen/qwen3.8-27b"),
]

if not api_key:
    print("FAILED: GROQ_API_KEY is missing from .env")
    raise SystemExit(1)

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
    timeout=60,
    max_retries=0,
)

question = """
Explain Retrieval-Augmented Generation (RAG) in simple terms.
Give one example of how it answers questions from a PDF.
"""

for model in models:
    print("\n" + "=" * 60)
    print("TESTING MODEL:", model)
    print("=" * 60)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. "
                        "Answer clearly and accurately."
                    ),
                },
                {"role": "user", "content": question},
            ],
            max_completion_tokens=700,
        )

        answer = response.choices[0].message.content

        print("STATUS: SUCCESS")
        print("MODEL:", response.model)
        print("FINISH REASON:", response.choices[0].finish_reason)
        print("ANSWER:")
        print(answer if answer else "No text returned.")
        print("USAGE:", response.usage)

    except Exception as error:
        print("STATUS: FAILED")
        print("ERROR TYPE:", type(error).__name__)
        print("ERROR DETAILS:", str(error)[:500])

print("\nBoth model tests completed.")
