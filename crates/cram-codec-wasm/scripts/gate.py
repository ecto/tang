#!/usr/bin/env python3
"""Headless browser gate: serve demo/, open index.html?gate&post in headless
Chrome (WebGPU on), wait for the page to POST its gate result, write it to
demo/gate.json and exit non-zero unless every stream is byte-identical.

    python3 scripts/gate.py [--chrome PATH] [--reps 3] [--port 8766] [--headed]
"""
import argparse, http.server, json, subprocess, sys, tempfile, threading
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent / "demo"
ap = argparse.ArgumentParser()
ap.add_argument("--chrome", default="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
ap.add_argument("--reps", default="3")
ap.add_argument("--port", type=int, default=8766)
ap.add_argument("--headed", action="store_true")
args = ap.parse_args()
done = threading.Event()
result = {}


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(DEMO), **k)

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(204)
        self.end_headers()
        if self.path == "/progress":
            print(body.decode(), flush=True)
            return
        result.update(json.loads(body))
        done.set()


srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{args.port}/index.html?gate&post&reps={args.reps}"
with tempfile.TemporaryDirectory() as prof:
    flags = [f"--user-data-dir={prof}", "--no-first-run", "--enable-unsafe-webgpu"]
    if not args.headed:
        flags.append("--headless=new")
    chrome = subprocess.Popen([args.chrome, *flags, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok = done.wait(timeout=1800)
    chrome.terminate()
    chrome.wait()
srv.shutdown()
if not ok:
    sys.exit("timed out waiting for the page")
(DEMO / "gate.json").write_text(json.dumps(result, indent=1))
s = result["summary"]
print(json.dumps(s, indent=1))
sys.exit(0 if s["identical"] == s["streams"] else 1)
