#!/usr/bin/env python3
"""dashboard.html を配り、画面の「更新」ボタンで収集をやり直す小さなサーバー。

標準ライブラリだけで動く。既定では、このサーバー自身（127.0.0.1）からしか開けない。
"""
import argparse
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "dashboard.html"
RUN_TIMEOUT_SECONDS = 120
LOCK = threading.Lock()


def refresh():
    """収集 → 画面の作り直し。同時に押されたら、順番に1回ずつ動かす。"""
    with LOCK:
        return subprocess.run([str(HERE / "run.sh")], timeout=RUN_TIMEOUT_SECONDS).returncode


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body, ctype="text/plain; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] not in ("/", "/dashboard.html"):
            return self.reply(404, b"not found")
        self.reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")

    def do_POST(self):
        if self.path != "/refresh":
            return self.reply(404, b"not found")
        # 他のサイトのページから勝手に呼ばれないよう、ボタンが付けるヘッダーを確かめる
        if self.headers.get("X-Refresh") != "1":
            return self.reply(403, b"forbidden")
        try:
            code = refresh()
        except subprocess.TimeoutExpired:
            return self.reply(500, "時間内に終わりませんでした".encode())
        if code != 0:
            return self.reply(500, f"run.sh が終了コード {code} で終わりました".encode())
        self.reply(200, b"ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    if not PAGE.exists():
        refresh()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"http://{args.host}:{args.port}/ で待ち受けます（Ctrl+C で止まる）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
