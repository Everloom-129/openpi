"""Vendored, dependency-light client for the GR00T inference server.

Mirrors gr00t/policy/server_client.py:PolicyClient but does NOT import the
`gr00t` package — that would pull in torch/transformers and is overkill for
the sim env. We only need to send msgpack-packed dicts over a ZMQ REQ socket
and decode the response (which contains numpy arrays serialized via np.save).

If the upstream protocol changes, sync this file with
third_party/Isaac-GR00T/gr00t/policy/server_client.py.
"""

from __future__ import annotations

import io
from typing import Any

import msgpack
import numpy as np
import zmq


def _encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj):
    if not isinstance(obj, dict):
        return obj
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    if "__ModalityConfig_class__" in obj:
        # Return the raw dict; we don't need the typed object on the client side.
        return obj["as_json"]
    return obj


def _to_bytes(data: Any) -> bytes:
    return msgpack.packb(data, default=_encode)


def _from_bytes(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_decode)


class PolicyClient:
    """Thin REQ/REP wrapper. One outstanding request at a time (matches server)."""

    def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 30000):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.ctx = zmq.Context.instance()
        self._open_socket()

    def _open_socket(self):
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
        req: dict = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data or {}
        try:
            self.sock.send(_to_bytes(req))
            msg = self.sock.recv()
        except zmq.error.Again:
            # REQ socket is wedged after a timeout; reopen.
            self.sock.close(0)
            self._open_socket()
            raise
        resp = _from_bytes(msg)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"GR00T server error: {resp['error']}")
        return resp

    def ping(self) -> dict:
        return self.call("ping", requires_input=False)

    def get_modality_config(self) -> dict:
        return self.call("get_modality_config", requires_input=False)

    def get_action(self, observation: dict, options: dict | None = None) -> tuple[dict, dict]:
        resp = self.call("get_action", {"observation": observation, "options": options})
        # Server returns (action_dict, info_dict) as a 2-tuple, msgpack flattens to list.
        action, info = resp[0], resp[1]
        return action, info
