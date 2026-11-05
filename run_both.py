"""Start both chatbots from ONE terminal.
Baseline (purple):      http://localhost:8501
Preprocessing (blue):   http://localhost:8502
Press Ctrl+C to stop both."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APPS = [("chatbot_1_baseline", 8501), ("chatbot_2_preprocessed", 8502)]


def main() -> None:
    procs = []
    try:
        for folder, port in APPS:
            # cwd = the chatbot's folder so its own .streamlit theme is used
            procs.append(subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "app.py",
                 "--server.port", str(port), "--server.headless", "true"],
                cwd=ROOT / folder))
        print("\nBaseline (purple):    http://localhost:8501")
        print("Preprocessing (blue): http://localhost:8502")
        print("Press Ctrl+C to stop both.\n")
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
