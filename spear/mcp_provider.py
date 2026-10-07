"""MCP as a capability provider: the stdio transport, tools only.

The part of the Model Context Protocol a provider needs here is small --
initialize, tools/list (paginated) and tools/call, as newline-delimited
JSON-RPC 2.0 over the server's stdin and stdout -- so it is implemented here
rather than through the reference SDK, whose client brings an ASGI server
stack and native cryptography with it. Server prompts and resources are not
requested: nothing a server offers reaches a turn except as a capability.

The server is started from its registered command, with only the
environment the registration names, and lives as long as the provider.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading

from capabilities import Capability, CapabilityError, Invocation, ProviderConfig, capability

PROTOCOL_VERSION = "2025-06-18"
CLIENT = {"name": "spear", "version": "1"}
_RESULT_CHARS = 20_000


class MCPProvider:
    """One registered MCP server, spoken to over stdio."""

    def __init__(self, config: ProviderConfig, *, cwd: str | None = None):
        self.config, self.id, self._cwd = config, config.id, cwd
        self._process = None
        self._lines: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._next = 0
        self._listed: list[Capability] | None = None

    # ── transport ───────────────────────────────────────────────────

    def _start(self):
        if self._process is not None and self._process.poll() is None:
            return

        self._listed = None
        environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                       "HOME": os.environ.get("HOME", "/"), **dict(self.config.env)}

        try:
            self._process = subprocess.Popen(
                list(self.config.command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, cwd=self._cwd, env=environment, text=True,
                encoding="utf-8", bufsize=1)
        except OSError as exc:
            raise CapabilityError(f"{self.id}: the server could not start ({exc})") from exc

        self._lines = queue.Queue()
        reader = threading.Thread(target=self._read, args=(self._process, self._lines),
                                  daemon=True)
        reader.start()
        self._request("initialize", {"protocolVersion": PROTOCOL_VERSION,
                                     "capabilities": {}, "clientInfo": CLIENT})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    @staticmethod
    def _read(process, lines):
        for line in process.stdout:
            lines.put(line)

        lines.put(None)

    def _send(self, message):
        try:
            self._process.stdin.write(json.dumps(message) + "\n")
            self._process.stdin.flush()
        except (OSError, ValueError) as exc:
            raise CapabilityError(f"{self.id}: the server is gone ({exc})") from exc

    def _request(self, method, params):
        with self._lock:
            self._next += 1
            request_id = self._next
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method,
                        "params": params})

            while True:
                try:
                    line = self._lines.get(timeout=self.config.timeout)
                except queue.Empty as exc:
                    self.close()
                    raise CapabilityError(f"{self.id}: no answer to {method} within "
                                          f"{self.config.timeout:g} s") from exc

                if line is None:
                    self.close()
                    raise CapabilityError(f"{self.id}: the server closed the connection")

                try:
                    message = json.loads(line)
                except ValueError:
                    continue                      # not a JSON-RPC message: logging

                if not isinstance(message, dict) or message.get("id") != request_id:
                    continue                      # a notification, or a stray answer

                if "error" in message:
                    error = message["error"] or {}
                    raise CapabilityError(f"{self.id}: {method} failed: "
                                          f"{str(error.get('message', error))[:300]}")

                result = message.get("result")

                if not isinstance(result, dict):
                    raise CapabilityError(f"{self.id}: {method} returned no result object")

                return result

    # ── the provider ────────────────────────────────────────────────

    def list(self) -> list[Capability]:
        """Every tool the server declares, as capabilities; cached until the
        connection is renewed. A tool whose declaration is unusable is left
        out; the rest stand."""
        if self._listed is not None and self._process is not None \
                and self._process.poll() is None:
            return list(self._listed)

        self._start()
        found, cursor, pages, self.skipped = [], None, 0, []

        while True:
            result = self._request("tools/list", {"cursor": cursor} if cursor else {})

            for tool in result.get("tools") or ():
                try:
                    found.append(capability(
                        self.config, str(tool["name"]), tool.get("description") or "",
                        tool.get("inputSchema"), provenance=f"mcp:{self.id}:tools/list"))
                except (KeyError, TypeError, CapabilityError) as exc:
                    self.skipped.append(str(exc))

            cursor, pages = result.get("nextCursor"), pages + 1

            if not cursor or pages >= 50:
                break

        self._listed = found

        return list(found)

    def invoke(self, name: str, arguments) -> Invocation:
        self._start()
        result = self._request("tools/call", {"name": name, "arguments": dict(arguments)})
        parts = []

        for block in result.get("content") or ():
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, dict):
                parts.append(f"[{block.get('type', 'content')} omitted]")

        if not parts and result.get("structuredContent") is not None:
            parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))

        text = "\n".join(parts)

        if len(text) > _RESULT_CHARS:
            text = text[:_RESULT_CHARS] + f"\n[... {len(text) - _RESULT_CHARS} chars omitted]"

        return Invocation(text, bool(result.get("isError")))

    def close(self) -> None:
        process, self._process, self._listed = self._process, None, None

        if process is None:
            return

        try:
            process.stdin.close()
        except (OSError, ValueError):
            pass

        try:
            process.terminate()
            process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            process.wait(timeout=3)

        try:
            process.stdout.close()
        except (OSError, ValueError):
            pass


def provider_for(config: ProviderConfig, **kwargs):
    if (config.protocol, config.transport) == ("mcp", "stdio"):
        return MCPProvider(config, **kwargs)

    raise CapabilityError(f"{config.id}: {config.protocol} over {config.transport} "
                          f"is not supported")
