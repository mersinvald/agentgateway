#!/usr/bin/env python3
"""Check Responses stream liveness through a real gateway and a local fake model.

No external model or API key is used. The upstream generates one tool call over
more than five minutes. The client enforces a shorter socket idle timeout.
"""

import argparse
import http.server
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", default="target/debug/agentgateway")
    parser.add_argument("--duration", type=float, default=330)
    parser.add_argument("--idle-timeout", type=float, default=25)
    parser.add_argument("--expect-timeout", action="store_true")
    args = parser.parse_args()
    assert args.duration > args.idle_timeout > 15
    gateway = str(Path(args.gateway).resolve())
    stop = threading.Event()
    errors = []
    arguments = json.dumps({"input": "quoted \"value\", slash \\, Unicode 世界"})

    class Backend(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            try:
                assert self.path == "/v1/chat/completions", self.path
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                name = request["tools"][0]["function"]["name"]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()

                def emit(delta, finish=None):
                    chunk = {"id": "chatcmpl_liveness", "object": "chat.completion.chunk",
                             "created": 1, "model": "gateway-smoke", "choices": [
                                 {"index": 0, "delta": delta, "finish_reason": finish}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()

                emit({"role": "assistant"})
                for index in range(10):
                    if stop.wait(args.duration / 11):
                        return
                    piece = arguments[index * len(arguments) // 10:(index + 1) * len(arguments) // 10]
                    tool = {"index": 0, "function": {"arguments": piece}}
                    if index == 0:
                        tool.update(id="call_liveness", type="function")
                        tool["function"]["name"] = name
                    emit({"reasoning_content": "progress ", "tool_calls": [tool]})
                if stop.wait(args.duration / 11):
                    return
                emit({}, "tool_calls")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                if not args.expect_timeout and not stop.is_set():
                    errors.append("unexpected upstream disconnect")
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")

    with tempfile.TemporaryDirectory(prefix="gateway-liveness-smoke-") as temp:
        directory = Path(temp)
        backend = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        threading.Thread(target=backend.serve_forever, daemon=True).start()
        port = free_port()
        config = {"config": {"adminAddr": "127.0.0.1:0", "readinessAddr": "127.0.0.1:0",
                             "statsAddr": "127.0.0.1:0"},
                  "llm": {"port": port, "models": [{"name": "gateway-smoke",
                      "provider": {"custom": {"formats": [{"type": "completions"}]}},
                      "params": {"baseUrl": f"http://127.0.0.1:{backend.server_port}/v1"}}]}}
        config_path = directory / "gateway.json"
        config_path.write_text(json.dumps(config))
        with (directory / "gateway.log").open("w+") as log:
            process = subprocess.Popen([gateway, "-f", str(config_path)], stdout=log, stderr=log)
            try:
                for _ in range(100):
                    if process.poll() is not None:
                        log.seek(0)
                        raise RuntimeError(log.read()[-4000:])
                    try:
                        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1).close()
                        break
                    except OSError:
                        time.sleep(0.1)
                else:
                    raise TimeoutError("Gateway did not start")
                payload = {"model": "gateway-smoke", "stream": True,
                           "input": "Emit the synthetic test tool call.", "tools": [
                               {"type": "function", "name": "write", "parameters": {
                                   "type": "object", "properties": {"input": {"type": "string"}},
                                   "required": ["input"]}}]}
                request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/responses",
                    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                started = last = last_report = time.monotonic()
                gaps, events, comments = [], [], 0
                timed_out = False
                try:
                    with urllib.request.urlopen(request, timeout=args.idle_timeout) as response:
                        for raw in response:
                            now = time.monotonic()
                            assert now - started < args.duration + 2 * args.idle_timeout, "stream did not terminate"
                            gaps.append(now - last)
                            last = now
                            line = raw.decode().strip()
                            if line.startswith(":"):
                                comments += 1
                            if now - last_report >= 60:
                                print(json.dumps({"status": "streaming", "elapsed_seconds": round(now - started, 1),
                                                  "heartbeats": comments}), flush=True)
                                last_report = now
                            if line.startswith("data: "):
                                event = json.loads(line[6:])
                                events.append(event)
                                assert event["type"] != "response.failed", event
                                if event["type"] == "response.output_item.done":
                                    assert now - started >= args.duration - 1, "premature tool completion"
                except (TimeoutError, socket.timeout):
                    timed_out = True
                elapsed = time.monotonic() - started
                assert timed_out == args.expect_timeout, {"timed_out": timed_out, "elapsed": elapsed}
                if not timed_out:
                    assert elapsed >= args.duration - 1, elapsed
                    assert comments > 0
                    assert max(gaps) < args.idle_timeout, max(gaps)
                    assert events[-1]["type"] == "response.completed", events[-1]
                    tools = [e["item"] for e in events if e["type"] == "response.output_item.done"
                             and e["item"]["type"] == "function_call"]
                    assert len(tools) == 1, tools
                    assert tools[0]["arguments"] == arguments, tools
                    assert tools[0]["name"] == "write", tools
                assert not errors, errors
                print(json.dumps({"result": "expected_timeout" if timed_out else "passed",
                                  "elapsed_seconds": round(elapsed, 3), "heartbeats": comments,
                                  "max_wire_gap_seconds": round(max(gaps, default=0), 3),
                                  "events": len(events)}), flush=True)
            finally:
                stop.set()
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                backend.shutdown()
                backend.server_close()


if __name__ == "__main__":
    main()
