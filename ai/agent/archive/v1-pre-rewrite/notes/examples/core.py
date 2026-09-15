"""Offline teaching components. Not a production runtime or provider SDK.
Python 3.11+; standard library only. Stored tool results are data, never commands.
"""
from __future__ import annotations
import codecs
import copy
import hashlib
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

class ProtocolError(ValueError):
    pass

def canonical(value: Any) -> str:
    # allow_nan=False rejects values JSON cannot represent portably.
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)

def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()

@dataclass(frozen=True)
class Tool:
    name: str
    revision: str
    parameters: tuple[str, ...]
    execute: Callable[[dict], Any]

class Registry:
    """Snapshots fix tool bindings, not permissions or nested mutable resources."""
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0
        self._tools: dict[str, Tool] = {}

    def install(self, tools: list[Tool]) -> int:
        if len({t.name for t in tools}) != len(tools):
            raise ValueError("duplicate tool names")
        with self._lock:
            self._generation += 1
            self._tools = {t.name: t for t in tools}
            return self._generation

    def snapshot(self) -> Mapping[str, Tool]:
        with self._lock:
            return MappingProxyType(self._tools.copy())

    def remove(self, generation: int) -> bool:
        with self._lock:
            if generation != self._generation:
                return False
            self._tools = {}
            self._generation += 1
            return True

class ToolGateway:
    """Exact-key validation only; examples use read-only deterministic tools."""
    def __init__(self, bindings: Mapping[str, Tool],
                 authorize: Callable[[str, dict], bool] | None = None):
        self.bindings = bindings
        self.authorize = authorize or (lambda name, args: True)
        self.effects = 0

    def execute(self, call: dict) -> dict:
        name, args = call.get("name"), call.get("arguments")
        tool = self.bindings.get(name) if isinstance(name, str) else None
        if tool is None:
            return {"status": "error", "code": "UNKNOWN_TOOL"}
        if not isinstance(args, dict) or set(args) != set(tool.parameters):
            return {"status": "error", "code": "INVALID_ARGUMENTS"}
        # Copy prevents the authorizer from modifying the arguments to execute.
        args = json.loads(canonical(args))
        if not self.authorize(name, copy.deepcopy(args)):
            return {"status": "denied", "code": "NOT_AUTHORIZED"}
        self.effects += 1
        try:
            value = tool.execute(args)
            return {"status": "ok", "value": json.loads(canonical(value))}
        except Exception as error:
            # Safe only for this lab's read-only tools. Writes need reconciliation.
            return {"status": "error", "code": type(error).__name__}

class ScriptedModel:
    """Records exact inputs so tests can inspect call/result correlation."""
    def __init__(self, replies: list[dict]):
        self.replies = copy.deepcopy(replies)
        self.requests: list[list[dict]] = []

    def complete(self, messages: list[dict], tools: Mapping[str, Tool]) -> dict:
        self.requests.append(copy.deepcopy(messages))
        if not self.replies:
            raise ProtocolError("script has no next reply")
        return self.replies.pop(0)

def normalize_reply(reply: dict) -> dict:
    if reply.get("complete") is not True:
        raise ProtocolError("incomplete response; no tool may execute")
    calls = reply.get("tool_calls", [])
    if not isinstance(calls, list):
        raise ProtocolError("tool_calls must be a list")
    ids = []
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not call["id"]:
            raise ProtocolError("missing call id")
        ids.append(call["id"])
    if len(set(ids)) != len(ids):
        raise ProtocolError("duplicate call ids")
    if reply.get("refusal") and calls:
        raise ProtocolError("ambiguous refusal and executable calls")
    return copy.deepcopy(reply)

def one_shot_tool(model: ScriptedModel, gateway: ToolGateway, query: str) -> dict:
    reply = normalize_reply(model.complete(
        [{"role": "user", "content": query}], gateway.bindings))
    if reply.get("refusal"):
        return {"status": "refused", "reason": reply["refusal"]}
    calls = reply.get("tool_calls", [])
    if len(calls) != 1:
        raise ProtocolError("one-shot endpoint requires exactly one tool")
    return gateway.execute(calls[0])  # No second model call.

def run_loop(model: ScriptedModel, gateway: ToolGateway, query: str,
             max_steps: int = 8, cancelled: Callable[[], bool] = lambda: False) -> dict:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    history = [{"role": "user", "content": query}]
    seen_ids: set[str] = set()
    for _ in range(max_steps):
        if cancelled():
            return {"status": "cancelled", "messages": history}
        reply = normalize_reply(model.complete(history, gateway.bindings))
        if reply.get("refusal"):
            return {"status": "refused", "messages": history}
        calls = reply.get("tool_calls", [])
        if seen_ids.intersection(c["id"] for c in calls):
            raise ProtocolError("call identity reused across rounds")
        seen_ids.update(c["id"] for c in calls)
        history.append({"role": "assistant", "content": reply.get("text", ""),
                        "tool_calls": copy.deepcopy(calls)})
        if not calls:
            return {"status": "answer_proposed", "answer": reply.get("text", ""),
                    "messages": history}
        stopped = False
        for call in calls:
            if cancelled():
                result = {"status": "not_started", "code": "CANCELLED_BEFORE_DISPATCH"}
                stopped = True
            else:
                result = gateway.execute(call)
            history.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        if stopped:
            return {"status": "cancelled", "messages": history}
    return {"status": "incomplete", "reason": "STEP_LIMIT", "messages": history}

def validate_pairs(messages: list[dict]) -> None:
    """Check a settled history; not a generic vendor wire validator."""
    pending: set[str] = set()
    seen: set[str] = set()
    for msg in messages:
        if msg["role"] == "assistant":
            if pending:
                raise ProtocolError("assistant precedes required results")
            for call in msg.get("tool_calls", []):
                if call["id"] in seen:
                    raise ProtocolError("duplicate call")
                pending.add(call["id"])
                seen.add(call["id"])
        elif msg["role"] == "tool":
            cid = msg["tool_call_id"]
            if cid not in pending:
                raise ProtocolError("orphan or duplicate result")
            pending.remove(cid)
        elif pending:
            raise ProtocolError("interleaved message before results")
    if pending:
        raise ProtocolError("missing results")

def select_groups(groups: list[dict], budget: int) -> list[dict]:
    """Token counts are supplied estimates. Required groups never silently drop."""
    if budget < 0 or any(g["tokens"] < 0 for g in groups):
        raise ValueError("negative budget or size")
    if len({g["id"] for g in groups}) != len(groups):
        raise ValueError("duplicate group")
    required = [g for g in groups if g.get("required")]
    if sum(g["tokens"] for g in required) > budget:
        raise ValueError("required context exceeds budget")
    chosen = {g["id"] for g in required}
    used = sum(g["tokens"] for g in required)
    candidates = sorted((g for g in groups if not g.get("required")),
                        key=lambda g: (-g.get("priority", 0), g["id"]))
    for group in candidates:
        if used + group["tokens"] <= budget:
            chosen.add(group["id"])
            used += group["tokens"]
    # Preserve source history order, not relevance-ranking order.
    return [copy.deepcopy(g) for g in groups if g["id"] in chosen]

def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    if k < 1:
        raise ValueError("k must be positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen = set()
        for rank, doc in enumerate(ranking, 1):
            if doc in seen:
                continue
            seen.add(doc)
            scores[doc] = scores.get(doc, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))

def require_coverage(expected: set[str], results: list[dict | None]) -> dict[str, dict]:
    found = {}
    for result in results:
        if result is None:
            continue
        key = result["task_id"]
        if key not in expected or key in found:
            raise ValueError("unexpected or duplicate task result")
        found[key] = result
    if set(found) != expected or any(r.get("verified") is not True for r in found.values()):
        raise ValueError("missing or unverified result")
    return found

class Budget:
    """Atomic teaching budget in integer units, not estimated vendor billing."""
    def __init__(self, limit: int):
        if limit < 0:
            raise ValueError("negative limit")
        self.limit = limit
        self.spent = 0
        self._reservations: dict[str, int] = {}
        self._lock = threading.Lock()

    def reserve(self, amount: int) -> str | None:
        if amount < 0:
            raise ValueError("negative reservation")
        with self._lock:
            if self.spent + sum(self._reservations.values()) + amount > self.limit:
                return None
            key = str(uuid.uuid4())
            self._reservations[key] = amount
            return key

    def settle(self, key: str, actual: int) -> None:
        if actual < 0:
            raise ValueError("negative consumption")
        with self._lock:
            if key not in self._reservations:
                raise ValueError("unknown or already settled reservation")
            # Overspend is recorded honestly, not clamped to the reservation.
            del self._reservations[key]
            self.spent += actual

    def release(self, key: str) -> None:
        with self._lock:
            if key not in self._reservations:
                raise ValueError("unknown reservation")
            del self._reservations[key]

class ApprovalStore:
    """One-shot approval identity. Caller provides current policy and tool version."""
    def __init__(self):
        self._entries: dict[str, str] = {}
        self._lock = threading.Lock()

    def issue(self, identity: dict) -> str:
        key = str(uuid.uuid4())
        with self._lock:
            self._entries[key] = fingerprint(identity)
        return key

    def consume(self, key: str, current_identity: dict) -> bool:
        with self._lock:
            expected = self._entries.get(key)
            if expected != fingerprint(current_identity):
                return False
            del self._entries[key]
            return True

class JsonLineStream:
    """Toy normalized JSONL protocol, NOT SSE or a vendor wire implementation.
    Preserves UTF-8 boundaries; item_done does not imply response_done.
    """
    def __init__(self):
        self._decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self._buffer = ""
        self._items: dict[str, dict] = {}
        self.done = False

    def feed(self, data: bytes) -> None:
        self._buffer += self._decoder.decode(data)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not line.strip():
                continue
            if self.done:
                raise ProtocolError("event after terminal response")
            event = json.loads(line)
            kind = event["type"]
            if kind == "start":
                key = event["id"]
                if key in self._items:
                    raise ProtocolError("duplicate item")
                self._items[key] = {"kind": event["kind"], "value": "", "done": False}
            elif kind == "delta":
                item = self._items[event["id"]]
                if item["done"]:
                    raise ProtocolError("delta after item end")
                item["value"] += event["value"]
            elif kind == "item_done":
                item = self._items[event["id"]]
                if item["done"]:
                    raise ProtocolError("duplicate item end")
                item["done"] = True
            elif kind == "response_done":
                if not all(item["done"] for item in self._items.values()):
                    raise ProtocolError("unfinished item")
                self.done = True
            else:
                raise ProtocolError("unknown required event")

    def finish(self) -> dict[str, dict]:
        self._buffer += self._decoder.decode(b"", final=True)
        if self._buffer.strip() or not self.done:
            raise ProtocolError("missing terminal or incomplete record")
        return copy.deepcopy(self._items)

@contextmanager
def _connect(path: Path):
    connection = sqlite3.connect(path, timeout=5.0)
    connection.execute("PRAGMA busy_timeout=5000")
    try:
        with connection:
            yield connection
    finally:
        connection.close()

class OperationLedger:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "ledger.sqlite"
        with _connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS ops"
                       "(id TEXT PRIMARY KEY, digest TEXT NOT NULL, state TEXT NOT NULL, receipt TEXT)")

    def prepare(self, operation: str, payload: dict) -> None:
        with _connect(self.path) as db:
            db.execute("INSERT OR IGNORE INTO ops VALUES (?, ?, 'prepared', NULL)",
                       (operation, fingerprint(payload)))
            row = db.execute("SELECT digest FROM ops WHERE id=?", (operation,)).fetchone()
            if row[0] != fingerprint(payload):
                raise ValueError("same operation id with different intent")

    def dispatched(self, operation: str) -> None:
        with _connect(self.path) as db:
            row = db.execute("UPDATE ops SET state='unknown' WHERE id=? AND state='prepared'",
                             (operation,))
            if row.rowcount != 1:
                raise ValueError("operation not prepared")

    def commit(self, operation: str, receipt: dict) -> None:
        with _connect(self.path) as db:
            row = db.execute("SELECT state, receipt FROM ops WHERE id=?", (operation,)).fetchone()
            if row is None:
                raise ValueError("unknown operation")
            encoded = canonical(receipt)
            if row[0] == "confirmed" and row[1] != encoded:
                raise ValueError("conflicting receipt")
            db.execute("UPDATE ops SET state='confirmed', receipt=? WHERE id=?",
                       (encoded, operation))

    def get(self, operation: str) -> tuple[str, dict | None]:
        with _connect(self.path) as db:
            row = db.execute("SELECT state, receipt FROM ops WHERE id=?", (operation,)).fetchone()
        if row is None:
            raise KeyError(operation)
        return row[0], json.loads(row[1]) if row[1] else None

class FakeRemote:
    """Separate SQLite file models a remote service with enforced idempotency.
    This contract is supplied by the simulator, not created by the local ledger.
    """
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "remote.sqlite"
        with _connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS effects"
                       "(id TEXT PRIMARY KEY, digest TEXT NOT NULL, receipt TEXT NOT NULL)")

    def apply(self, operation: str, payload: dict) -> dict:
        receipt = {"operation_id": operation, "object_id": "draft-" + operation}
        with _connect(self.path) as db:
            db.execute("INSERT OR IGNORE INTO effects VALUES (?, ?, ?)",
                       (operation, fingerprint(payload), canonical(receipt)))
            row = db.execute("SELECT digest, receipt FROM effects WHERE id=?", (operation,)).fetchone()
            if row[0] != fingerprint(payload):
                raise ValueError("remote idempotency conflict")
            return json.loads(row[1])

    def query(self, operation: str) -> dict | None:
        with _connect(self.path) as db:
            row = db.execute("SELECT receipt FROM effects WHERE id=?", (operation,)).fetchone()
        return json.loads(row[0]) if row else None

    def count(self) -> int:
        with _connect(self.path) as db:
            return db.execute("SELECT COUNT(*) FROM effects").fetchone()[0]

def recover(ledger: OperationLedger, remote: FakeRemote, operation: str) -> dict:
    state, receipt = ledger.get(operation)
    if state == "confirmed":
        return receipt
    observed = remote.query(operation)
    if observed is None:
        # Absence is NOT treated as permission to repeat an arbitrary write.
        return {"status": "needs_reconciliation"}
    ledger.commit(operation, observed)
    return observed
