"""Optional live API example. Not run in validation; may incur provider charges.
Install the official `openai` Python package separately. Set OPENAI_API_KEY
and OPENAI_MODEL to values available to your account. SDK version is user-pinned.
"""
import os
from openai import OpenAI

def main():
    model = os.environ.get("OPENAI_MODEL")
    if not model:
        raise SystemExit("Set OPENAI_MODEL; no default model is assumed.")
    client = OpenAI(timeout=30.0, max_retries=0)
    response = client.responses.create(
        model=model,
        instructions="解释用户提供的配置，区分直接观察与推测。",
        input="server:\n  port: 8080",
    )
    if response.status != "completed":
        raise RuntimeError(f"Incomplete response: {response.status}")
    print(response.output_text)

if __name__ == "__main__":
    main()
