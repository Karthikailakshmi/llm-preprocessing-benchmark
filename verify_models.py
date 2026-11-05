"""Stage 1: list the model IDs each provider exposes to YOUR API key."""
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

PROVIDERS = {
    "gemini": ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/models", "flash"),
    "nvidia": ("NVIDIA_API_KEY", "https://integrate.api.nvidia.com/v1/models", "nemotron"),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/models", ":free"),
}


def openrouter_limits(key: str) -> None:
    try:
        resp = requests.get("https://openrouter.ai/api/v1/key",
                            headers={"Authorization": f"Bearer {key}"}, timeout=20)
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            for field in ("is_free_tier", "limit", "limit_remaining", "usage"):
                print(f"    key.{field}: {data.get(field)}")
        else:
            print(f"    key info: HTTP {resp.status_code}")
    except requests.RequestException as exc:
        print(f"    key info: NETWORK ERROR {type(exc).__name__}")


def main() -> None:
    show_all = "--all" in sys.argv
    # Hugging Face model check
    hf_token = os.getenv("HF_TOKEN")

    if hf_token:
        try:
            hf_url = "https://huggingface.co/api/models/openai/gpt-oss-120b"
            hf_response = requests.get(
                hf_url,
                headers={"Authorization": f"Bearer {hf_token}"},
                timeout=20,
            )

            if hf_response.status_code == 200:
                model_info = hf_response.json()
                print(
                    f"[huggingface] MODEL FOUND: "
                    f"{model_info.get('id', 'openai/gpt-oss-120b')}"
                )
            else:
                print(
                    f"[huggingface] HTTP {hf_response.status_code} "
                    "(check model access or token)"
                )

        except requests.RequestException as exc:
            print(f"[huggingface] NETWORK ERROR: {type(exc).__name__}")
    else:
        print("[huggingface] SKIPPED: HF_TOKEN not set in .env")

    for name, (env_var, url, keyword) in PROVIDERS.items():
        key = os.getenv(env_var)
        if not key:
            print(f"[{name}] SKIPPED: {env_var} not set in .env")
            continue
        try:
            resp = requests.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=20)
        except requests.RequestException as exc:
            print(f"[{name}] NETWORK ERROR: {type(exc).__name__}")
            continue
        if resp.status_code != 200:
            print(f"[{name}] HTTP {resp.status_code} (check key, quota, or billing)")
            continue
        ids = sorted(m.get("id", "").removeprefix("models/") for m in resp.json().get("data", []))
        shown = ids if show_all else [i for i in ids if keyword in i.lower()]
        print(f"[{name}] OK - {len(ids)} models, showing {len(shown)}:")
        for model_id in shown:
            print(f"    {model_id}")
        if name == "openrouter":
            openrouter_limits(key)


if __name__ == "__main__":
    main()
