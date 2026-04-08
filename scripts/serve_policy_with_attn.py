"""Policy server with real-time attention visualization.

Hosts a pi05_droid_torch WebSocket policy server that captures attention
on every inference and serves a live visualization on a separate HTTP port.

Architecture:
  - WebSocket (port 8000): robot client sends obs, receives actions
  - HTTP     (port 8001): browser auto-refreshes to show latest attention

The action is returned to the robot as soon as inference + attention capture
finishes. The matplotlib rendering runs in a background thread so it does
not block the control loop.

Usage:
    uv run python scripts/serve_policy_with_attn.py \
        --config pi05_droid \
        --checkpoint-dir checkpoints/viz/pi05_droid_pytorch \
        --device cuda:0

    # Then open http://localhost:8001 in a browser
    # and connect your robot client to ws://localhost:8000
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import http.server
import io
import json
import logging
import socket
import threading
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ── Shared state between policy server and HTTP viz server ───────────────────

class AttnState:
    """Thread-safe container for the latest attention + images."""

    def __init__(self):
        self._lock = threading.Lock()
        self._step = 0
        self._data: dict | None = None
        # PNG bytes of the latest visualization
        self._png_bytes: bytes = b""
        # Background renderer thread pool (1 thread — renders sequentially,
        # never blocks the inference/action return path).
        self._render_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="viz")

    def update(self, data: dict, png_bytes: bytes):
        with self._lock:
            self._step += 1
            self._data = data
            self._png_bytes = png_bytes

    def submit_render(self, attn_data: dict, viz_layer: int, viz_head: str):
        """Submit a render job to the background thread. Non-blocking."""
        self._render_pool.submit(self._render_and_update, attn_data, viz_layer, viz_head)

    def _render_and_update(self, attn_data: dict, viz_layer: int, viz_head: str):
        """Render PNG and update shared state. Runs in background thread."""
        try:
            t0 = time.monotonic()
            png = render_attention_png(attn_data, layer=viz_layer, head=viz_head)
            viz_ms = (time.monotonic() - t0) * 1000
            self.update(attn_data, png)
            logger.info(
                "Step %d rendered: viz=%.0fms, infer was %.0fms",
                self.step, viz_ms, attn_data["infer_ms"],
            )
        except Exception:
            logger.exception("Background render failed")

    @property
    def step(self) -> int:
        with self._lock:
            return self._step

    @property
    def png(self) -> bytes:
        with self._lock:
            return self._png_bytes

    @property
    def data(self) -> dict | None:
        with self._lock:
            return self._data


ATTN_STATE = AttnState()

# ── Visualization ────────────────────────────────────────────────────────────

NUM_IMAGE_TOKENS = 256
TOTAL_IMAGE_TOKENS = 512
TEXT_START_IDX = 768
PATCH_GRID = 16


def _upsample_heatmap(attn_512: np.ndarray, camera: str) -> np.ndarray:
    """16×16 patch attention → 224×224 heatmap."""
    offset = 0 if camera == "exterior" else NUM_IMAGE_TOKENS
    patches = attn_512[offset : offset + NUM_IMAGE_TOKENS]
    grid = patches.reshape(PATCH_GRID, PATCH_GRID).astype(np.float32)
    return np.kron(grid, np.ones((14, 14), dtype=np.float32))


def render_attention_png(
    attn_data: dict,
    layer: int = 10,
    head: str = "mean",
) -> bytes:
    """Render attention overlay as PNG bytes.

    Shows:
      Row 1: text→image attention overlay on exterior + wrist cameras
              for the top-5 attending text tokens.
      Row 2: action→image attention overlay (suffix) if available.
    """
    meta = attn_data["meta"]
    images = attn_data["images"]
    prefix = attn_data["prefix"]
    joint = attn_data.get("joint")
    token_texts = meta["token_texts"]
    n_text = meta["n_real_tokens"]

    layer_key = f"layer_{layer}"
    t2i = prefix[layer_key]["text_to_img"]  # (8, n_text, 512)

    # Aggregate heads
    if head == "mean":
        t2i_agg = t2i.mean(axis=0)  # (n_text, 512)
    elif head == "max":
        t2i_agg = t2i.max(axis=0)
    else:
        t2i_agg = t2i[int(head)]

    # Find top-5 text tokens by total image attention
    token_attn_sum = t2i_agg[:, :TOTAL_IMAGE_TOKENS].sum(axis=1)
    top_k = min(5, n_text)
    top_idxs = np.argsort(token_attn_sum)[-top_k:][::-1]

    has_joint = joint is not None and layer_key in joint
    n_rows = 2 if has_joint else 1
    fig, axes = plt.subplots(n_rows, 2 * top_k, figsize=(4 * top_k, 4 * n_rows), squeeze=False)

    ext_img = images.get("exterior")
    wrist_img = images.get("wrist")

    for col_i, tok_idx in enumerate(top_idxs):
        tok_label = token_texts[tok_idx] if tok_idx < len(token_texts) else f"tok_{tok_idx}"
        attn_vec = t2i_agg[tok_idx]  # (512,)

        for cam_i, (cam_name, cam_img) in enumerate([("exterior", ext_img), ("wrist", wrist_img)]):
            ax = axes[0, col_i * 2 + cam_i]
            hmap = _upsample_heatmap(attn_vec, cam_name)
            if cam_img is not None:
                ax.imshow(cam_img)
                ax.imshow(hmap, cmap="hot", alpha=0.5, vmin=0, vmax=hmap.max() + 1e-8)
            else:
                ax.imshow(hmap, cmap="hot")
            ax.set_title(f"{tok_label} → {cam_name}", fontsize=8)
            ax.axis("off")

    # Row 2: action → image attention
    if has_joint:
        a2i = joint[layer_key]["action_to_img"]  # (8, 8, 512)
        a2i_mean = a2i.mean(axis=0)  # (8, 512) — average over heads
        # Show action steps 0, 2, 4, 6 (or as many as top_k allows)
        action_steps = list(range(0, 8, max(1, 8 // top_k)))[:top_k]
        for col_i, step in enumerate(action_steps):
            attn_vec = a2i_mean[step]
            for cam_i, (cam_name, cam_img) in enumerate([("exterior", ext_img), ("wrist", wrist_img)]):
                ax = axes[1, col_i * 2 + cam_i]
                hmap = _upsample_heatmap(attn_vec, cam_name)
                if cam_img is not None:
                    ax.imshow(cam_img)
                    ax.imshow(hmap, cmap="hot", alpha=0.5, vmin=0, vmax=hmap.max() + 1e-8)
                else:
                    ax.imshow(hmap, cmap="hot")
                ax.set_title(f"action[{step}] → {cam_name}", fontsize=8)
                ax.axis("off")

    fig.suptitle(
        f"Step {ATTN_STATE.step} | Layer {layer} | Head: {head}\n"
        f"Instruction: {meta['instruction'][:80]}",
        fontsize=10,
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ── HTTP visualization server ────────────────────────────────────────────────

INDEX_HTML = """\
<!DOCTYPE html>
<html>
<head>
  <title>Attention Visualizer</title>
  <style>
    body { background: #1a1a2e; color: #eee; font-family: monospace; text-align: center; margin: 20px; }
    img { max-width: 95vw; border: 1px solid #444; border-radius: 4px; }
    .info { margin: 10px; font-size: 14px; color: #aaa; }
    #status { color: #0f0; }
  </style>
</head>
<body>
  <h2>Pi0.5 Real-Time Attention</h2>
  <div class="info">Step: <span id="status">waiting...</span></div>
  <div><img id="attn" src="/attn.png" /></div>
  <div class="info" id="meta"></div>
  <script>
    let lastStep = -1;
    async function poll() {
      try {
        const r = await fetch('/status.json');
        const d = await r.json();
        if (d.step !== lastStep) {
          lastStep = d.step;
          document.getElementById('status').textContent = `Step ${d.step} (${d.infer_ms} ms)`;
          document.getElementById('attn').src = '/attn.png?t=' + Date.now();
          document.getElementById('meta').textContent = d.instruction || '';
        }
      } catch(e) {}
      setTimeout(poll, 200);
    }
    poll();
  </script>
</body>
</html>
"""


class VizHandler(http.server.BaseHTTPRequestHandler):
    """Serves the live attention visualization page."""

    def log_message(self, format, *args):
        pass  # suppress per-request logs

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._respond(200, "text/html", INDEX_HTML.encode())
        elif self.path.startswith("/attn.png"):
            png = ATTN_STATE.png
            if png:
                self._respond(200, "image/png", png)
            else:
                self.send_error(204, "No attention data yet")
        elif self.path == "/status.json":
            data = ATTN_STATE.data
            info = {
                "step": ATTN_STATE.step,
                "infer_ms": round(data["infer_ms"], 1) if data else 0,
                "instruction": data["meta"]["instruction"] if data else "",
            }
            self._respond(200, "application/json", json.dumps(info).encode())
        else:
            self.send_error(404)

    def _respond(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


def start_viz_server(port: int):
    """Run the HTTP viz server in a daemon thread."""
    server = http.server.HTTPServer(("0.0.0.0", port), VizHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    logger.info("Attention visualizer running at http://0.0.0.0:%d", port)
    return server


# ── Policy server with attention capture ─────────────────────────────────────

def run_inference_with_attn(policy, obs: dict, viz_layer: int, viz_head: str) -> dict:
    """Infer + capture attention, submit viz to background, return actions immediately.

    The matplotlib rendering happens in a background thread so the robot
    gets its action back without waiting for the PNG to be drawn.
    """
    from openpi.models_pytorch import gemma_pytorch as _gpt

    _gpt.enable_attn_buffer()
    _gpt.enable_suffix_attn_buffer()
    try:
        t0 = time.monotonic()
        result = policy.infer(obs)
        infer_ms = (time.monotonic() - t0) * 1000

        buf = _gpt.get_attn_buffer()
        suffix_buf = _gpt.get_suffix_attn_buffer()
    finally:
        _gpt.clear_attn_buffer()
        _gpt.clear_suffix_attn_buffer()

    if not buf:
        logger.warning("Attention buffer empty — skipping visualization")
        return result

    # ── Build attention data dict (same schema as inference.py) ──────────
    first = next(iter(buf.values()))
    seq_len = int(first.shape[-1])
    n_text = seq_len - TEXT_START_IDX

    # Token labels
    token_texts = [f"tok_{i}" for i in range(n_text)]
    n_text_actual = n_text
    try:
        from openpi.models.tokenizer import PaligemmaTokenizer

        inputs_copy = {**obs}
        transformed = policy._input_transform(inputs_copy)
        token_ids = np.asarray(transformed["tokenized_prompt"])
        token_mask = np.asarray(transformed["tokenized_prompt_mask"])
        n_real = int(token_mask.sum())
        real_ids = token_ids[:n_real].tolist()
        tokenizer = PaligemmaTokenizer()
        token_texts = [tokenizer._tokenizer.id_to_piece(i) for i in real_ids]
        n_text_actual = min(n_text, len(token_texts))
    except Exception as e:
        logger.warning("Tokenizer decode failed: %s", e)
        n_text_actual = min(n_text, len(token_texts))

    # Resize images to 224x224
    def _to_224(img):
        if img is None:
            return None
        from PIL import Image as _PIL
        return np.array(_PIL.fromarray(img.astype(np.uint8)).resize((224, 224), _PIL.BILINEAR), dtype=np.uint8)

    # Build prefix dict
    prefix = {}
    for layer_idx, attn in buf.items():
        if attn.ndim == 4:
            attn = attn[0]
        attn = attn.astype(np.float32)
        t2i = attn[:, TEXT_START_IDX : TEXT_START_IDX + n_text_actual, :TOTAL_IMAGE_TOKENS]
        prefix[f"layer_{layer_idx}"] = {"text_to_img": t2i, "full": attn}

    # Build joint (suffix) dict
    joint = {}
    if suffix_buf:
        for layer_idx, sa in suffix_buf.items():
            if sa.ndim == 4:
                sa = sa[0]
            sa = sa.astype(np.float32)
            a2i = sa[:, :, :TOTAL_IMAGE_TOKENS]
            a2t = sa[:, :, TEXT_START_IDX : TEXT_START_IDX + n_text_actual]
            a2a = sa[:, :, seq_len:]
            joint[f"layer_{layer_idx}"] = {
                "action_to_img": a2i,
                "action_to_text": a2t,
                "action_to_action": a2a,
            }

    attn_data = {
        "meta": {
            "prefix_len": TEXT_START_IDX,
            "seq_len": seq_len,
            "n_real_tokens": n_text_actual,
            "instruction": obs.get("prompt", ""),
            "token_texts": token_texts[:n_text_actual],
        },
        "images": {
            "exterior": _to_224(obs.get("observation/exterior_image_1_left")),
            "wrist": _to_224(obs.get("observation/wrist_image_left")),
        },
        "prefix": prefix,
        "joint": joint or None,
        "pred_action": result.get("actions"),
        "infer_ms": infer_ms,
    }

    # ── Submit rendering to background thread (non-blocking) ─────────────
    ATTN_STATE.submit_render(attn_data, viz_layer, viz_head)

    return result


class AttnPolicyServer:
    """WebSocket policy server that captures and visualizes attention on every step."""

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
        viz_layer: int = 10,
        viz_head: str = "mean",
    ):
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._viz_layer = viz_layer
        self._viz_head = viz_head

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        import websockets.asyncio.server as _server

        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket):
        from openpi_client import msgpack_numpy

        logger.info("Connection from %s", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                # Synchronous: infer + viz before replying
                action = await asyncio.get_event_loop().run_in_executor(
                    None,
                    run_inference_with_attn,
                    self._policy,
                    obs,
                    self._viz_layer,
                    self._viz_head,
                )

                action["server_timing"] = {
                    "infer_ms": (time.monotonic() - start) * 1000,
                }
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start

            except Exception as e:
                if "ConnectionClosed" in type(e).__name__:
                    logger.info("Connection closed: %s", websocket.remote_address)
                    break
                logger.exception("Handler error")
                await websocket.send(traceback.format_exc())
                await websocket.close(code=1011, reason="Internal error")
                raise

    @staticmethod
    async def _health_check(connection, request):
        if request.path == "/healthz":
            return connection.respond(200, "OK\n")
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Pi0.5 policy server with real-time attention viz")
    p.add_argument("--config", default="pi05_droid", help="Training config name")
    p.add_argument("--checkpoint-dir", required=True, help="Path to PyTorch checkpoint dir")
    p.add_argument("--device", default="cuda:0", help="PyTorch device")
    p.add_argument("--port", type=int, default=8000, help="WebSocket policy port")
    p.add_argument("--viz-port", type=int, default=8001, help="HTTP visualization port")
    p.add_argument("--viz-layer", type=int, default=10, help="Attention layer to visualize (0-17)")
    p.add_argument("--viz-head", default="mean", help="Head aggregation: 'mean', 'max', or 0-7")
    p.add_argument("--default-prompt", default=None, help="Fallback prompt if not in obs")
    return p.parse_args()


def main():
    args = parse_args()

    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config

    # Load model
    logger.info("Loading %s from %s on %s ...", args.config, args.checkpoint_dir, args.device)
    config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(
        config, args.checkpoint_dir,
        pytorch_device=args.device,
        default_prompt=args.default_prompt,
    )
    logger.info("Model loaded.")

    # Start HTTP viz server (background thread)
    start_viz_server(args.viz_port)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info("Policy WebSocket: ws://%s:%d", local_ip, args.port)
    logger.info("Attention viewer: http://%s:%d", local_ip, args.viz_port)

    # Start WebSocket policy server (blocks)
    server = AttnPolicyServer(
        policy=policy,
        port=args.port,
        metadata=policy.metadata,
        viz_layer=args.viz_layer,
        viz_head=args.viz_head,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s %(levelname)s %(message)s")
    main()
