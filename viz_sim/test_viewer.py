"""Minimal smoke test: launch a robocasa kitchen scene with the interactive
on-screen MuJoCo viewer.

Run inside the robocasa_sim conda env:
    conda activate robocasa_sim
    python viz_sim/test_viewer.py

Requires a working display ($DISPLAY set). Over SSH that means either X11
forwarding (`ssh -Y`) or a virtual display (xvfb / VirtualGL). See
viz_sim/setup_robocasa_env.sh comments and the snippets at the bottom of
this file's docstring.
"""

import os

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config

import robocasa  # noqa: F401  (registers envs)


def main():
    if not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "DISPLAY is not set. Over SSH, reconnect with `ssh -Y user@host` "
            "or start a virtual display (see viz_sim/README)."
        )

    controller_config = load_composite_controller_config(robot="PandaOmron")
    env = robosuite.make(
        env_name="PickPlaceCounterToCabinet",
        robots="PandaOmron",
        controller_configs=controller_config,
        has_renderer=True,            # on-screen GLFW window
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera="robot0_agentview_center",
        ignore_done=True,
        control_freq=20,
    )

    env.reset()
    print("Scene loaded. Rendering 500 idle frames — close the window to exit.")
    for _ in range(500):
        env.step(np.zeros(env.action_dim))
        env.render()
    env.close()


if __name__ == "__main__":
    main()
