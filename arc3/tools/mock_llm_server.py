"""Minimal OpenAI-compatible chat server for offline harness smoke tests.

Every request is answered with one `python` tool call that inspects the frame
and plays a random valid action, so the full Duck harness loop (prompting,
sandbox, action execution, level bookkeeping) runs without a GPU.
"""
import json
import random
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CODE = """
import random
segs = current_frame.segmentation
print('objects', len(segs['nodes']), 'valid', valid_actions)
acts = [a for a in valid_actions if a != 'RESET'] or valid_actions
a = random.choice(acts)
if a == 'MOUSE':
    r = action([{'action': 'MOUSE', 'row': random.randrange(64), 'col': random.randrange(64)}])
else:
    r = action([a])
print('changed', r['action_result'].get('board_changed') if isinstance(r, dict) and 'action_result' in r else r)
"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": "mock", "object": "model"}]})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        msg = {
            "role": "assistant",
            "content": "World model: mock.\nPlan: random probe.",
            "tool_calls": [{
                "id": "call_%d" % random.randrange(1 << 30),
                "type": "function",
                "function": {"name": "python", "arguments": json.dumps({"code": CODE})},
            }],
        }
        if not req.get("tools"):
            msg = {"role": "assistant", "content": "ok"}
        self._send({
            "id": "mock", "object": "chat.completion", "model": req.get("model", "mock"),
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        })


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18234
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
