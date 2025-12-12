import json
import os
import shutil
import subprocess

EVAL_JSON = "Evaluation/eval_videommmu.json"
COOKIE_FILE = os.environ.get(
    "YT_COOKIES",
    "/mnt/ubuntudata/CSE589VisLan/ytcookies_netscape.txt",  # Netscape-format cookies
)
SLEEP_REQ_MIN = int(os.environ.get("YT_SLEEP_MIN", "10"))
# Pick a JS runtime to silence yt-dlp EJS warning
JS_RUNTIME = os.environ.get("YT_JS_RUNTIME")
if JS_RUNTIME is None:
    if shutil.which("node"):
        JS_RUNTIME = "node"
    elif shutil.which("nodejs"):  # some distros name it nodejs
        JS_RUNTIME = "nodejs"

with open(EVAL_JSON, "r") as f:
    data = json.load(f)

for i, item in enumerate(data):
    rel_path = item["path"]             # e.g. "./Evaluation/VideoMMMU/dev_Biology_3.mp4"
    url = item["data_source"]           # Youtube url like "https://www.youtube.com/watch?v=..."

    out_path = rel_path.lstrip("./")    # -> "Evaluation/VideoMMMU/dev_Biology_3.mp4"

    # Make sure directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Skip if already downloaded
    if os.path.exists(out_path):
        print(f"[{i}] already exists, skip: {out_path}")
        continue

    print(f"[{i}] downloading\n  url:  {url}\n  ->   {out_path}")

    # Use yt-dlp to download as mp4
    cmd = [
        "yt-dlp",
        "-f", "mp4",
        "--sleep-requests", str(SLEEP_REQ_MIN),  # pause between requests to avoid rate limits
        "--limit-rate", "1.5M",      # throttle bandwidth to look less bot-like
        "--retries", "5",
        "--fragment-retries", "5",
        "-N", "1",                   # single connection per file
        "-o", out_path,
    ]

    if COOKIE_FILE and os.path.exists(COOKIE_FILE):
        cmd.extend(["--cookies", COOKIE_FILE])

    if JS_RUNTIME:
        cmd.extend(["--js-runtimes", JS_RUNTIME])

    cmd.append(url)

    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"[{i}] yt-dlp failed for {url}: {e}")
