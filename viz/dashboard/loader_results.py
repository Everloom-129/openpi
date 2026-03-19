"""Loader for RESULTS_ROOT benchmark directory structure.

RESULTS_ROOT/
└── {success,failure}/
    └── {date}/
        └── {episode_id}/
            ├── pi05.md              ← completion marker
            ├── 00000/
            │   ├── 00000.h5         ← main inference
            │   ├── 00000_cube.h5    ← counterfactual variant
            │   └── ...
            ├── 00008/
            └── ...

Reuses cached loaders from loader.py (load_meta, load_images, etc.)
so HDF5 reads stay cached across the Results and Offline modes.
"""
from __future__ import annotations

from pathlib import Path


def list_outcomes(root: str) -> list[str]:
    r = Path(root)
    if not r.exists():
        return []
    return sorted(
        d.name for d in r.iterdir()
        if d.is_dir() and d.name in ("success", "failure")
    )


def list_dates(root: str, outcome: str) -> list[str]:
    d = Path(root) / outcome
    if not d.exists():
        return []
    return sorted((x.name for x in d.iterdir() if x.is_dir()), reverse=True)


def list_episodes(root: str, outcome: str, date: str) -> list[str]:
    d = Path(root) / outcome / date
    if not d.exists():
        return []
    return sorted(x.name for x in d.iterdir() if x.is_dir())


def list_frames(root: str, outcome: str, date: str, episode: str) -> list[int]:
    ep_dir = Path(root) / outcome / date / episode
    if not ep_dir.exists():
        return []
    frames = []
    for d in sorted(ep_dir.iterdir()):
        if d.is_dir():
            try:
                frames.append(int(d.name))
            except ValueError:
                pass
    return frames


def list_cf_slugs(
    root: str, outcome: str, date: str, episode: str, frame: int
) -> list[str]:
    """List counterfactual variant slugs for a given frame (e.g. ['cube','pen','empty'])."""
    frame_dir = Path(root) / outcome / date / episode / f"{frame:05d}"
    if not frame_dir.exists():
        return []
    prefix = f"{frame:05d}_"
    slugs = []
    for f in sorted(frame_dir.glob(f"{prefix}*.h5")):
        slug = f.stem[len(prefix):]
        if slug:
            slugs.append(slug)
    return slugs


def is_complete(root: str, outcome: str, date: str, episode: str) -> bool:
    """Return True if pi05.md completion marker exists for this episode."""
    return (Path(root) / outcome / date / episode / "pi05.md").exists()


def h5_path_results(
    root: str,
    outcome: str,
    date: str,
    episode: str,
    frame: int,
    slug: str | None = None,
) -> str:
    """Resolve path to an HDF5 file in the results directory.

    slug=None  → {frame:05d}/{frame:05d}.h5        (main inference)
    slug="cube"→ {frame:05d}/{frame:05d}_cube.h5   (counterfactual)
    """
    frame_dir = Path(root) / outcome / date / episode / f"{frame:05d}"
    if slug:
        return str(frame_dir / f"{frame:05d}_{slug}.h5")
    return str(frame_dir / f"{frame:05d}.h5")
