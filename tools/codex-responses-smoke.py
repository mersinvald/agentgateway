#!/usr/bin/env python3
"""Exercise a real Codex client against AgentGateway and a local deterministic backend.

No API key or external model is used. The only tool action creates smoke.txt in
an automatically removed temporary workspace.
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
    parser.add_argument("--codex", default="codex")
    args = parser.parse_args()
    gateway = str(Path(args.gateway).resolve())
    state = {"requests": 0, "tool_result": False, "shell_result": False, "errors": []}

    class Backend(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            try:
                assert self.path == "/v1/chat/completions", self.path
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                state["requests"] += 1
                results = [m for m in body["messages"] if m["role"] == "tool"]
                tools = [t["function"] for t in body["tools"]]
                if any(m["tool_call_id"] == "call_smoke_shell" for m in results):
                    assert any("protocol-ok" in m["content"] for m in results if m["tool_call_id"] == "call_smoke_shell")
                    state["shell_result"] = True
                    delta, finish = {"content": "protocol-ok"}, "stop"
                elif results:
                    assert any(m["tool_call_id"] == "call_smoke" for m in results)
                    assert any(m.get("content") == "I'll create the file." for m in body["messages"] if m["role"] == "assistant")
                    state["tool_result"] = True
                    shell = next(t for t in tools if t["name"] == "exec_command" or "Tool: functions.exec_command" in t.get("description", ""))
                    delta = {"content": "I'll verify the file.", "tool_calls": [{"index": 0, "id": "call_smoke_shell", "type": "function",
                             "function": {"name": shell["name"], "arguments": json.dumps({"cmd": "cat smoke.txt", "max_output_tokens": 100})}}]}
                    finish = "tool_calls"
                else:
                    state["tools"] = [{"name": t["name"], "properties": list(t.get("parameters", {}).get("properties", {}))} for t in tools]
                    patch = next(t for t in tools if "apply_patch" in t["name"]
                                 or "Apply a patch" in t.get("description", "")
                                 or "*** Begin Patch" in t.get("description", ""))
                    assert "input" in patch["parameters"]["properties"], patch["name"]
                    raw = "*** Begin Patch\n*** Add File: smoke.txt\n+protocol-ok\n*** End Patch"
                    delta = {"content": "I'll create the file.", "tool_calls": [{"index": 0, "id": "call_smoke", "type": "function",
                             "function": {"name": patch["name"], "arguments": json.dumps({"input": raw})}}]}
                    finish = "tool_calls"
                chunk = {"id": "chatcmpl_smoke", "object": "chat.completion.chunk", "created": 1,
                         "model": "gateway-smoke", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                payload = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as exc:
                state["errors"].append(f"{type(exc).__name__}: {exc}")
                self.send_error(500, "Mock protocol assertion failed")

    with tempfile.TemporaryDirectory(prefix="codex-gateway-smoke-") as temp:
        workspace = Path(temp)
        backend = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        threading.Thread(target=backend.serve_forever, daemon=True).start()
        port = free_port()
        config = {"config": {"adminAddr": "127.0.0.1:0", "readinessAddr": "127.0.0.1:0", "statsAddr": "127.0.0.1:0"},
                  "llm": {"port": port, "models": [{"name": "gateway-smoke",
                      "provider": {"custom": {"formats": [{"type": "completions"}]}},
                      "params": {"baseUrl": f"http://127.0.0.1:{backend.server_port}/v1"}}]}}
        config_path = workspace / "gateway.json"
        config_path.write_text(json.dumps(config))
        catalog = json.loads((Path(__file__).resolve().parents[1] / "examples/codex-kimi/models.json").read_text())
        catalog["models"][0]["slug"] = "gateway-smoke"
        catalog_path = workspace / "models.json"
        catalog_path.write_text(json.dumps(catalog))
        with (workspace / "gateway.log").open("w+") as log:
            process = subprocess.Popen([gateway, "-f", str(config_path)], stdout=log, stderr=log)
            try:
                for _ in range(100):
                    if process.poll() is not None:
                        log.seek(0)
                        raise RuntimeError(log.read())
                    try:
                        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1).close()
                        break
                    except OSError:
                        time.sleep(0.1)
                command = [args.codex, "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
                           "-C", temp, "-s", "workspace-write", "--json", "-m", "gateway-smoke",
                           "-c", 'model_provider="gateway_smoke"',
                           "-c", f'model_providers.gateway_smoke={{name="Local smoke backend",base_url="http://127.0.0.1:{port}/v1",wire_api="responses",requires_openai_auth=false,request_max_retries=0,stream_max_retries=0}}',
                           "-c", f'model_catalog_json="{catalog_path}"',
                           "-c", 'web_search="disabled"',
                           "-c", 'model_reasoning_effort="none"',
                           "Create smoke.txt containing protocol-ok using apply_patch, read it with a shell command, then reply protocol-ok."]
                result = subprocess.run(command, capture_output=True, text=True, timeout=90, stdin=subprocess.DEVNULL)
                assert not state["errors"], state["errors"]
                if result.returncode:
                    log.seek(0)
                    raise AssertionError(json.dumps(state) + "\n" + log.read()[-6000:] + "\n" + result.stderr + result.stdout)
                assert state["tool_result"], result.stdout + result.stderr
                assert state["shell_result"], result.stdout + result.stderr
                assert (workspace / "smoke.txt").read_text().strip() == "protocol-ok"
                print(json.dumps({"result": "passed", "requests": state["requests"],
                                  "custom_tool_executed": True, "tool_result_replayed": True, "shell_result_replayed": True}))
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                backend.shutdown()


if __name__ == "__main__":
    main()
