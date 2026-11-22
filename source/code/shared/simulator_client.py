"""Small, serial Python client for the local official simulator interface.

This module does not start a test or select a search strategy.  Its CLI prints
help unless a caller explicitly builds a client and invokes its methods.
"""

import argparse
import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


class SimulatorClientError(RuntimeError):
    """Base class for client-side protocol and state errors."""


class TransportFailure(SimulatorClientError):
    """No trustworthy HTTP response was received; remote execution is unknown."""


class HttpStatusFailure(SimulatorClientError):
    def __init__(self, status, body):
        super().__init__(f"HTTP {status}: {body}")
        self.status, self.body = status, body


class ActionRejected(SimulatorClientError):
    """The simulator answered HTTP 200 but accepted=false."""


class StateUncertain(SimulatorClientError):
    """Retry the pending request or explicitly exit before new stateful actions."""


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: dict


class LocalHttpTransport:
    """HTTP transport pinned to local loopback and with all proxies disabled."""

    def __init__(self, port=2026, timeout_s=5.0):
        self.base_url = f"http://127.0.0.1:{int(port)}"
        self.timeout_s = float(timeout_s)
        self._opener = build_opener(ProxyHandler({}))

    def post(self, path, payload):
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = Request(self.base_url + path, data=encoded,
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
                body = json.loads(raw)
                if not isinstance(body, dict):
                    raise ValueError('Response JSON must be an object')
                return HttpResult(response.status, body)
        except HTTPError as error:
            raw = error.read().decode("utf-8", errors="replace")
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = {"raw_body": raw}
            raise HttpStatusFailure(error.code, body) from error
        except (URLError, TimeoutError, ConnectionError, OSError, ValueError, UnicodeError) as error:
            raise TransportFailure(str(error)) from error


class SimulatorClient:
    """Serial stateful wrapper for /enter, /measure, /clear and /exit."""

    def __init__(self, robot_id, *, transport=None, port=2026, log_path=None, exit_reserve_s=5.0):
        if not isinstance(robot_id, str) or not robot_id:
            raise ValueError("robot_id must be a non-empty string")
        self.robot_id = robot_id
        self.transport = transport or LocalHttpTransport(port=port)
        if log_path is False:
            self.log_path = None
        elif log_path is None:
            root = Path(__file__).resolve().parents[2]
            stamp = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            self.log_path = root / "results" / "simulator_logs" / (stamp + ".jsonl")
        else:
            self.log_path = Path(log_path)
        self.exit_reserve_s = float(exit_reserve_s)
        self.entered = False
        self.ended = False
        self.position = None
        self.channel = None
        self.virtual_time_s = None
        self._entered_at = None
        self._enter_requested_at = None
        self._remaining_at_enter_s = None
        self._pending = None
        self._uncertain = False
        self._lock = threading.RLock()

    @property
    def uncertain(self):
        return self._uncertain

    @property
    def pending_request(self):
        return None if self._pending is None else dict(self._pending[1])

    def remaining_real_time_s(self):
        if self._entered_at is None or self._remaining_at_enter_s is None:
            return None
        return max(0.0, self._remaining_at_enter_s - (time.monotonic() - self._entered_at))

    def _new_payload(self, request_id=None, **fields):
        return {"arena_id": "default", "robot_id": self.robot_id,
                "request_id": request_id or uuid.uuid4().hex, **fields}

    def _write_log(self, record):
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _ensure_new_action_allowed(self, path):
        if self._pending is not None:
            raise StateUncertain("A request is pending; call retry_pending() or force_exit().")
        if self._uncertain and path in ("/measure", "/clear"):
            raise StateUncertain("Remote state is uncertain; do not send a new measure/clear action.")
        if path in ("/measure", "/clear"):
            if not self.entered or self.ended:
                raise SimulatorClientError("measure/clear requires a successfully entered, active test")
            remaining = self.remaining_real_time_s()
            request_budget = getattr(self.transport, 'timeout_s', 0.0)
            if remaining is not None and remaining <= self.exit_reserve_s + request_budget:
                raise SimulatorClientError("exit reserve reached; do not start another stateful action")

    def _apply_accepted(self, path, payload, body):
        # Only accepted=true responses update locally tracked simulator state.
        self.virtual_time_s = body.get("virtual_time_s", self.virtual_time_s)
        if path == "/enter":
            self.entered, self.ended = True, False
            self.position, self.channel = (0.0, 0.0), 1
            self._remaining_at_enter_s = float(body.get("remaining_real_duration_s", 0.0))
            # A cached /enter response must not restart the 20-minute clock.
            self._entered_at = self._enter_requested_at
        elif path == "/measure":
            self.position = (payload["position"]["x"], payload["position"]["y"])
            self.channel = payload["channel"]
        elif path == "/clear":
            self.position = (payload["position"]["x"], payload["position"]["y"])
            # Deliberately do not change self.channel: /clear never switches it.
        elif path == "/exit":
            self.ended = True

    def _send(self, path, payload, *, is_retry=False):
        record = {"path": path, "request": payload, "retry": is_retry}
        try:
            result = self.transport.post(path, payload)
            record["http_status"] = result.status
            record["response"] = result.body
        except TransportFailure as error:
            record["transport_failure"] = str(error)
            self._pending, self._uncertain = (path, payload), True
            self._write_log(record)
            raise
        except HttpStatusFailure as error:
            record["http_status"], record["http_error"] = error.status, error.body
            # 5xx can leave execution unclear; 4xx is a syntactically rejected request.
            if error.status >= 500:
                self._pending, self._uncertain = (path, payload), True
            self._write_log(record)
            raise
        self._write_log(record)
        if result.status != 200:
            raise HttpStatusFailure(result.status, result.body)
        if result.body.get("accepted") is not True:
            # In particular, virtual_time_s=0 here must never overwrite local time.
            raise ActionRejected(f"{path} was rejected: {result.body}")
        self._apply_accepted(path, payload, result.body)
        self._pending, self._uncertain = None, False
        return result.body

    def enter(self):
        with self._lock:
            if self.entered and not self.ended:
                raise SimulatorClientError("already entered")
            self._ensure_new_action_allowed("/enter")
            self._enter_requested_at = time.monotonic()
            return self._send("/enter", self._new_payload())

    def measure(self, x, y, channel):
        with self._lock:
            self._ensure_new_action_allowed("/measure")
            return self._send("/measure", self._new_payload(position={"x": x, "y": y}, channel=channel))

    def clear(self, x, y, channel):
        with self._lock:
            self._ensure_new_action_allowed("/clear")
            return self._send("/clear", self._new_payload(position={"x": x, "y": y}, channel=channel))

    def exit(self):
        """Normal exit; unavailable while a prior request is pending."""
        with self._lock:
            if not self.entered or self.ended:
                raise SimulatorClientError("exit requires an active entered test")
            self._ensure_new_action_allowed("/exit")
            return self._send("/exit", self._new_payload())

    def retry_pending(self):
        """Retry exactly the same uncertain request, including request_id and JSON body."""
        with self._lock:
            if self._pending is None:
                raise SimulatorClientError("no pending request to retry")
            return self._send(self._pending[0], self._pending[1], is_retry=True)

    def force_exit(self):
        """Explicit recovery route after an uncertain action; never sends measure/clear."""
        with self._lock:
            if not self.entered or self.ended:
                raise SimulatorClientError("force_exit requires an active entered test")
            return self._send("/exit", self._new_payload())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-id", help="Current logged-in team number; no request is sent by this CLI.")
    parser.add_argument("--port", type=int, default=2026)
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
