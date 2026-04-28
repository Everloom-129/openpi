"""Aggregate the perturbation experiment into results/report.md.

Reads /mnt/sda/edward/projects/robocasa_365_perturb/{task}/{condition}/ep_*.npz
and emits per-task tables + per-condition aggregate. Statistics:
  - success rate, mean max-reward (paired Wilcoxon vs baseline)
  - eef trajectory deviation from baseline (mean L2 over time, matched seed)
  - executed-action deviation from baseline (mean L2 over time, matched seed)
"""
from __future__ import annotations
import json
import pathlib
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
PERT_DIR = pathlib.Path("/mnt/sda/edward/projects/robocasa_365_perturb")
OUT_PATH = REPO / "results" / "report.md"

TASKS = ["Lift", "Stack", "Door", "PickPlaceCan", "NutAssemblySquare"]
CONDITIONS = [
    "baseline",
    "ext_zero_max", "ext_strengthen_max", "ext_zero_min",
    "wrist_zero_max", "wrist_strengthen_max", "wrist_zero_min",
]


def load_eps(task, cond):
    d = PERT_DIR / task / cond
    rows = []
    for p in sorted(d.glob("ep_*.npz")):
        try:
            x = np.load(p, allow_pickle=True)
        except Exception:
            continue
        rows.append({
            "ep": int(p.stem.split("_")[1]),
            "seed": int(x["seed"]) if "seed" in x.files else -1,
            "success": bool(x["success"]),
            "max_reward": float(x["max_reward"]),
            "final_reward": float(x["final_reward"]),
            "steps": int(x["steps"]),
            "eef_traj": x["eef_traj"] if "eef_traj" in x.files else None,
            "action_taken": x["action_taken"] if "action_taken" in x.files else None,
        })
    return rows


def paired_wilcoxon(a, b):
    """Tiny self-contained Wilcoxon signed-rank (two-sided). Returns (W, p)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    diff = a - b
    diff = diff[diff != 0]
    if len(diff) < 2:
        return 0.0, 1.0
    abs_d = np.abs(diff)
    order = np.argsort(abs_d)
    ranks = np.empty_like(order, dtype=float)
    # average ranks for ties
    sorted_abs = abs_d[order]
    i = 0
    n = len(sorted_abs)
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j + 1)
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    signs = np.sign(diff)
    W_plus = float(np.sum(ranks[signs > 0]))
    W_minus = float(np.sum(ranks[signs < 0]))
    W = min(W_plus, W_minus)
    # normal approx (n>=10 OK; small-n is conservative)
    mu = n * (n + 1) / 4.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return W, 1.0
    z = (W - mu) / sigma
    # two-sided p via erfc
    from math import erfc, sqrt
    p = erfc(abs(z) / sqrt(2))
    return W, float(p)


def deviation_from_baseline(perturbed_rows, baseline_rows, key):
    """Per-seed mean L2 distance over time. Returns array of per-episode means."""
    base_by_seed = {r["seed"]: r for r in baseline_rows}
    devs = []
    for r in perturbed_rows:
        b = base_by_seed.get(r["seed"])
        if b is None or r[key] is None or b[key] is None:
            continue
        a = np.asarray(r[key]); c = np.asarray(b[key])
        T = min(len(a), len(c))
        if T == 0: continue
        d = np.linalg.norm(a[:T] - c[:T], axis=-1).mean()
        devs.append(float(d))
    return np.asarray(devs, dtype=float)


def fmt_p(p):
    if p < 0.001: return "<0.001"
    return f"{p:.3f}"


def render_task_section(task: str) -> str:
    base = load_eps(task, "baseline")
    if not base:
        return f"\n## {task}\n\n_No baseline data._\n"
    base_max = np.array([r["max_reward"] for r in base], dtype=float)

    lines = [f"\n## {task}", "", f"_Episodes per condition: baseline n={len(base)}_", ""]
    lines.append("| condition | n | succ % | mean rmax | Δrmax vs base | p (Wilcoxon) | mean Δeef-L2 | mean Δaction-L2 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cond in CONDITIONS:
        rows = load_eps(task, cond) if cond != "baseline" else base
        if not rows:
            lines.append(f"| {cond} | 0 | — | — | — | — | — | — |"); continue
        n = len(rows)
        succ = sum(r["success"] for r in rows)
        rmax = np.array([r["max_reward"] for r in rows], dtype=float)
        mean_rmax = float(rmax.mean())
        if cond == "baseline":
            delta_rmax = "—"; pval = "—"; eef_dev = "—"; act_dev = "—"
        else:
            paired_base = np.array([r["max_reward"] for r in base if r["seed"] in {x["seed"] for x in rows}], dtype=float)
            paired_per  = np.array([r["max_reward"] for r in rows if r["seed"] in {x["seed"] for x in base}], dtype=float)
            delta_rmax = f"{paired_per.mean() - paired_base.mean():+.3f}"
            _, p = paired_wilcoxon(paired_per, paired_base)
            pval = fmt_p(p)
            eef = deviation_from_baseline(rows, base, "eef_traj")
            act = deviation_from_baseline(rows, base, "action_taken")
            eef_dev = f"{eef.mean():.4f}" if eef.size else "—"
            act_dev = f"{act.mean():.4f}" if act.size else "—"
        lines.append(f"| {cond} | {n} | {100*succ/n:.1f} | {mean_rmax:.3f} | {delta_rmax} | {pval} | {eef_dev} | {act_dev} |")
    return "\n".join(lines) + "\n"


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = []
    out.append("# Layer-7 attention-perturbation experiment — pi05_robocasa365\n")
    out.append("## Hypothesis\n")
    out.append(
        "If the action expert relies on the image patches that the prefix attends to, "
        "then masking the **highest-attention** patch (per camera) on layer 7 should "
        "*degrade* policy behavior more than masking the **lowest-attention** patch. "
        "Conversely, **strengthening** the highest-attention patch should leave behavior "
        "intact or slightly improve it (already in-distribution salience).\n"
    )
    out.append("## Method\n")
    out.append(
        "- **Model**: pi05_robocasa365 (config=pi05_droid).\n"
        "- **Tasks**: Lift, Stack, Door, PickPlaceCan, NutAssemblySquare (robosuite Panda, JOINT_VELOCITY).\n"
        "- **Conditions** (7 each): baseline + {ext, wrist} × {zero_max, strengthen_max(×1.5), zero_min}.\n"
        "- **Intervention**: between prefix and suffix forward in `gemma_pytorch.py`, identify "
        "argmax/argmin image position from layer-7 text→image attention (head-mean × text-row-mean), "
        "then scale `past_key_values.value_cache[7][..., pos, :]` by the configured factor.\n"
        "- **Matched seeds**: each condition runs the same env-seed sequence (`seed_base + ep`) so "
        "rollouts are paired across conditions.\n"
        "- **Metrics**:\n"
        "  - **succ %**: episodes with reward > 0.5 at any step.\n"
        "  - **rmax**: max reward over rollout.\n"
        "  - **Δeef-L2**: per-step L2 distance between perturbed and baseline `robot0_eef_pos`, averaged over time and matched eps.\n"
        "  - **Δaction-L2**: same for `action_taken` (8-D after gripper binarization).\n"
        "  - **p**: paired Wilcoxon signed-rank on `rmax` vs baseline.\n"
    )
    out.append("## Results per task")
    for t in TASKS:
        out.append(render_task_section(t))

    # Aggregate across tasks
    out.append("\n## Aggregate across tasks\n")
    out.append("| condition | n_eps | mean rmax | Δrmax vs base | mean Δeef-L2 | mean Δaction-L2 |")
    out.append("|---|---|---|---|---|---|")
    base_all = []
    for t in TASKS:
        base_all.extend(load_eps(t, "baseline"))
    for cond in CONDITIONS:
        rmax_all, eef_all, act_all = [], [], []
        for t in TASKS:
            base = load_eps(t, "baseline")
            rows = load_eps(t, cond) if cond != "baseline" else base
            if not rows: continue
            rmax_all.extend(r["max_reward"] for r in rows)
            if cond != "baseline":
                e = deviation_from_baseline(rows, base, "eef_traj")
                a = deviation_from_baseline(rows, base, "action_taken")
                eef_all.extend(e.tolist()); act_all.extend(a.tolist())
        n = len(rmax_all)
        if n == 0:
            out.append(f"| {cond} | 0 | — | — | — | — |"); continue
        rmax_arr = np.asarray(rmax_all)
        if cond == "baseline":
            delta = "—"
        else:
            base_rmax = np.asarray([r["max_reward"] for r in base_all])
            delta = f"{rmax_arr.mean() - base_rmax.mean():+.3f}"
        eef = f"{np.mean(eef_all):.4f}" if eef_all else "—"
        act = f"{np.mean(act_all):.4f}" if act_all else "—"
        out.append(f"| {cond} | {n} | {rmax_arr.mean():.3f} | {delta} | {eef} | {act} |")

    out.append("\n## Conclusion (auto-generated stub)\n")
    out.append(
        "Inspect the per-task tables above. A meaningful effect of the hypothesis would show:\n"
        "  - `ext_zero_max` and `wrist_zero_max` having **larger** Δaction-L2 than `*_zero_min`.\n"
        "  - `*_strengthen_max` having **smaller** Δaction-L2 than `*_zero_max`.\n"
        "  - Optional: success/rmax shifts (likely small at this near-zero baseline).\n"
    )
    out.append("\nVideos: `results/video_perturb/{task}.webp` (7 condition columns, layer-7 attention overlay).\n")

    OUT_PATH.write_text("\n".join(out))
    print(f"[report] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
