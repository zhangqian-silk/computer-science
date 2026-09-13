"""Controlled failure in a temporary directory. Never calls a real service."""
import os
import sys
from pathlib import Path
from core import FakeRemote, OperationLedger

directory = Path(sys.argv[1])
ledger = OperationLedger(directory)
remote = FakeRemote(directory)
payload = {"title": "Review configuration"}
ledger.prepare("op-1", payload)
ledger.dispatched("op-1")
remote.apply("op-1", payload)   # Remote simulator commits first.
os._exit(17)                   # Deliberately skip the local receipt write.
