You were right to be suspicious. There's a major action-space mismatch for at least 2 of
  the 3 checkpoints — and the experiment currently running is using the wrong controller for
  all three.

  Per-checkpoint specs

  pi05_droid

  - Norm stats (first 8 active dims): action std ≈ [0.16, 0.31, 0.15, 0.30, 0.23, 0.25, 0.27,
   0.44], action mean ≈ [0, 0.011, 0.002, 0.028, 0.001, -0.003, 0.003, 0.45]
  - Action interpretation (per src/openpi/training/droid_rlds_dataset.py:34):
  ▎ "We default to joint position actions, since they allow policy evaluation in simulation."
  - So action[0:7] are delta joint position in radians (small per-step deltas; means ≈ 0);
  action[7] is gripper ∈ [0,1] (0=open, 1=close), DROID convention.
  - NOT joint velocity. DROID's collected actions in raw RLDS contain both joint_velocity and
   joint_position; openpi trains on joint_position by default.
  - State: [joint_pos(7), gripper(1)] — 8 dims.

  pi05_libero

  - Norm stats: action dim=7 (no padding!), std [0.33, 0.37, 0.45, 0.04, 0.06, 0.07, 0.99],
  mean [0.03, 0.09, -0.10, 0, 0, 0, -0.13]
  - Action interpretation: [Δeef_pos(3), Δeef_axisangle(3), gripper(1)] — OSC_POSE delta with
   gripper ∈ [-1, +1] (Libero/robosuite convention).
  - State: [eef_pos(3), eef_axisangle(3), gripper_qpos(2)] — 8 dims.

  pi05_robocasa365  ← most broken case

  - Norm stats: 31 active dims out of 32, first 12: std [0.37, 0.32, 0.33, 0.085, 0.10, 0.10,
   0.99, 0.23, 0.19, 0.15, 0.0, 0.70], mean [0.006, -0.004, -0.04, 0.002, -0.01, 0.001,
  -0.17, 0.04, 0.004, 0.001, 0, -0.72]
  - Action interpretation (matches third_party/robocasa/robocasa/utils/env_utils.py:134
  convert_action):
  action[0:3]   eef_pos delta        (cartesian xyz)
  action[3:6]   eef_rotation delta   (axis-angle)
  action[6]     gripper_close        (∈ [-1, +1])
  action[7:11]  base_motion          (4-D mobile base!)
  action[11]    control_mode         (arm vs base switch)
  action[12:31] additional active dims — likely a second arm or proprio joint commands
  - Trained on PandaOmron (Panda on a 4-DoF mobile base), NOT a fixed-base Panda.
  - State: also 30 active dims (eef + axisangle + gripper_qpos + joint_pos + base + ...).

  What our eval is actually doing wrong

  Our eval_perturb.py / eval_runner.py (per viz_sim/eval_perturb.py:64-87):
  1. Builds obs in DROID format (observation/joint_position, observation/gripper_position).
  2. Uses robosuite JOINT_VELOCITY controller (output_max=0.5).
  3. Slices actions[0:8] and feeds to env, with gripper binarization at 0.5.

  Mapping mismatches:

  ckpt: pi05_droid
  model output dims 0-6 are…: Δjoint_position (rad)
  we feed them as…: joint velocity (rad/s)
  result: systematic drift; magnitudes off — sometimes coincidentally OK because both are
    small
  ────────────────────────────────────────
  ckpt: pi05_libero
  model output dims 0-6 are…: [Δeef_pos, Δeef_axisangle] (cartesian + rot)
  we feed them as…: joint velocity for joints 1-7
  result: wrong axes; arm flails
  ────────────────────────────────────────
  ckpt: pi05_robocasa365
  model output dims 0-6 are…: [Δeef_pos, Δeef_axisangle, gripper, base, …] (cartesian + rot +

    base)
  we feed them as…: joint velocity for joints 1-7
  result: dims 3-5 (eef rotation deltas) drive joints 4-6 → wrist roll/pitch joints get
    rotation-magnitude commands as velocities → "gripper rotates too much"

  Gripper sign also matters: DROID is [0,1], Libero/robocasa are [-1,+1]. Our adapt_action
  (eval_perturb.py:72) does 1 if a[-1]>0.5 else -1 — correct for DROID, wrong for
  Libero/robocasa where the model already outputs ±1.

  Implications for the running experiment

  The currently-running per-task-horizon perturbation experiment will produce data, but it's
  data on a broken eval. The Δeef-L2/Δaction-L2 deltas from the perturbation are real (the
  model still attends to images), but the absolute trajectories are garbage because we're
  commanding robocasa-format cartesian deltas as joint velocities. Should I stop it?

  What's the right fix

  For meaningful evals, we'd need separate sim adapters per ckpt:

  ckpt: pi05_droid
  robosuite controller: JOINT_POSITION (target abs) or JOINT_VELOCITY if we explicitly load
    DROID's velocity-action variant
  obs builder: make_droid_obs (current)
  gripper map: binarize >0.5 → +1
  env: fixed-base Panda ✓
  ────────────────────────────────────────
  ckpt: pi05_libero
  robosuite controller: OSC_POSE (delta cartesian)
  obs builder: make_libero_obs (state = [eef_pos, eef_axisangle, gripper_qpos])
  gripper map: passthrough (already ±1)
  env: fixed-base Panda + Libero scene
  ────────────────────────────────────────
  ckpt: pi05_robocasa365
  robosuite controller: PandaOmron + composite (eef + base + gripper + control_mode) as in
    robocasa's env_utils.py
  obs builder: robocasa obs (3 cams + state.eef_9d + base + joint + gripper)
  gripper map: passthrough
  env: PandaOmron in robocasa kitchen env, NOT the robosuite Panda Lift/Stack/etc tasks

  Robocasa365 is not really evaluable on robosuite Panda Lift/Stack/Door/etc — those tasks
  don't exist in robocasa's eval suite, and the action space is entirely different. The
  natural eval for robocasa365 is robocasa target-task suite (50 kitchen tasks listed in
  dataset_registry.py).

  Suggested next steps

  Pick one:
  - (A) Stop the run; switch the experiment to pi05_libero on Libero's own task suite
  (Libero-Spatial/Libero-Object/Libero-Goal/Libero-10) using OSC_POSE. The libero ckpt in our
   assets/physical-intelligence/libero/norm_stats.json matches that.
  - (B) Stop the run; eval pi05_droid only with a JOINT_POSITION controller in robosuite.
  That's the smallest-diff fix and keeps the same Panda task suite.
  - (C) Keep the run going as a "perturbation effect under wrong-controller" sanity check
  (the relative perturbation effects are still meaningful even if absolute behavior is
  garbage), and separately set up the correct eval afterwards. Honest but messy.
  - (D) Setup full robocasa eval for pi05_robocasa365 (PandaOmron + composite controller +   kitchen tasks). Biggest scope, most valid.

  Which direction do you want?