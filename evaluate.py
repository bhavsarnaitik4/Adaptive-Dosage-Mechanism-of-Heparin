import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from physiology import get_true_aptt
from ecmo_env import ECMOHeparinEnv

# ==========================================
# EVALUATION CONSTANTS
# ==========================================
STRICT_CLOT_BURDEN  = 4.0    # External mortality threshold (stricter than env's 15.0)
STRICT_BLEED_APTT   = 110.0  # External hemorrhage threshold
MISSED_LAB_PROB     = 0.35   # 35% of lab draws are missed/delayed


# ==========================================
# 1. REALISTIC NOMOGRAM
# ==========================================
def realistic_clinical_nomogram(current_aptt, current_dose, weight):
    """
    Weight-based composite nomogram (Toronto General / Cincinnati Children's).
    Returns new dose rate (U/kg/hr) — weight-normalized so large patients
    get appropriately larger absolute adjustments.
    """
    if current_aptt < 50.0:
        rate_delta = +4.0
    elif current_aptt < 60.0:
        rate_delta = +2.0
    elif current_aptt <= 80.0:
        rate_delta = 0.0       # Hold — in therapeutic range
    elif current_aptt <= 100.0:
        rate_delta = -2.0
    elif current_aptt <= 120.0:
        rate_delta = -4.0
    else:
        rate_delta = -6.0

    return float(np.clip(current_dose + rate_delta, 5.0, 35.0))


def _apply_nomogram_dose(env, new_dose):
    """
    Injects an externally calculated dose into the env, then steps with a
    zero-delta action so the env's internal physics runs but doesn't override
    the dose we set.  action[0]=0 → delta=0 → current_dose unchanged.
    """
    env.current_dose = new_dose
    obs, reward, terminated, truncated, info = env.step(
        np.array([0.0], dtype=np.float32)
    )
    return obs, terminated, truncated


# ==========================================
# 2. METRIC CALCULATION
# ==========================================
def calculate_metrics(aptt_history, dose_history):
    """
    Calculates all clinical and operational metrics for one episode.
    No trimming of data — every step including terminal crashes is included.
    """
    aptt_arr = np.array(aptt_history, dtype=np.float64)
    dose_arr = np.array(dose_history, dtype=np.float64)

    if len(aptt_arr) == 0:
        return dict(ttt=120, auc=0.0, chatter=0.0, adjustments=0,
                    cv=0.0, dangerous_hours=0, peak_aptt=0.0, trough_aptt=0.0)

    # Time To Target (steps × 4 hours per step → hours)
    in_range_idx = np.where((aptt_arr >= 60.0) & (aptt_arr <= 80.0))[0]
    ttt = int(in_range_idx[0]) * 4 if len(in_range_idx) > 0 else 120

    # AUC Error: total distance from therapeutic zone over whole episode
    auc_error = float(np.sum(
        np.maximum(0.0, 60.0 - aptt_arr) + np.maximum(0.0, aptt_arr - 80.0)
    ))

    # Dose stability
    dose_diffs = np.abs(np.diff(dose_arr))
    chatter      = float(np.mean(dose_diffs))      if len(dose_diffs) > 0 else 0.0
    adjustments  = int(np.sum(dose_diffs > 0.05))  if len(dose_diffs) > 0 else 0
    mean_dose    = float(np.mean(dose_arr))
    cv           = float(np.std(dose_arr) / mean_dose * 100) if mean_dose > 0 else 0.0

    # Safety markers
    dangerous_hours = int(np.sum(aptt_arr < 50.0))   # Severe sub-therapeutic
    peak_aptt       = float(np.max(aptt_arr))
    trough_aptt     = float(np.min(aptt_arr))

    return dict(ttt=ttt, auc=auc_error, chatter=chatter,
                adjustments=adjustments, cv=cv,
                dangerous_hours=dangerous_hours,
                peak_aptt=peak_aptt, trough_aptt=trough_aptt)


# ==========================================
# 3. NOMOGRAM EVALUATION LOOP
# ==========================================
def evaluate_nomogram(env, episodes=1000):
    keys = ["ttr_list", "ttt", "auc", "chatter", "adjustments",
            "cv", "dangerous_hours", "peak_aptt", "trough_aptt"]
    metrics = {k: [] for k in keys}
    survived = fatal_clots = fatal_bleeds = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        done    = False
        aptt_hist, dose_hist = [], []
        hours_in_target = 0
        ep_outcome = "survived"

        while not done:
            # Decide dose (with missed-lab noise)
            if np.random.random() >= MISSED_LAB_PROB:
                new_dose = realistic_clinical_nomogram(
                    obs[0], env.current_dose, env.patient_weight
                )
            else:
                new_dose = env.current_dose  # Hold current drip rate

            obs, terminated, truncated = _apply_nomogram_dose(env, new_dose)

            # ── FIX: Record aPTT BEFORE checking strict thresholds ──
            true_aptt = get_true_aptt(env.state[0], env.state[1])
            aptt_hist.append(true_aptt)
            dose_hist.append(env.current_dose)
            if 60.0 <= true_aptt <= 80.0:
                hours_in_target += 1

            # External mortality checks (stricter than env's terminal conditions)
            if env.state[2] > STRICT_CLOT_BURDEN:
                ep_outcome = "clot"
                done = True
            elif env.last_lab_aptt > STRICT_BLEED_APTT:
                ep_outcome = "bleed"
                done = True
            elif terminated or truncated:
                ep_outcome = "survived"
                done = True

        if ep_outcome == "survived":
            survived += 1
        elif ep_outcome == "clot":
            fatal_clots += 1
        else:
            fatal_bleeds += 1

        if len(aptt_hist) > 0:
            ttr = (hours_in_target / len(aptt_hist)) * 100.0
            metrics["ttr_list"].append(ttr)
            m = calculate_metrics(aptt_hist, dose_hist)
            for k in keys[1:]:
                metrics[k].append(m[k])

    metrics["survived"]     = survived
    metrics["clots"]        = fatal_clots
    metrics["bleeds"]       = fatal_bleeds
    metrics["total"]        = episodes
    return metrics


# ==========================================
# 4. RL AGENT EVALUATION LOOP
# ==========================================
def evaluate_rl(eval_env, model, episodes=1000):
    keys = ["ttr_list", "ttt", "auc", "chatter", "adjustments",
            "cv", "dangerous_hours", "peak_aptt", "trough_aptt"]
    metrics = {k: [] for k in keys}
    survived = fatal_clots = fatal_bleeds = 0
    base_env = eval_env.venv.envs[0]

    for ep in range(episodes):
        obs = eval_env.reset()
        done = False
        aptt_hist, dose_hist = [], []
        hours_in_target = 0
        ep_outcome = "survived"

        while not done:
            # Missed-lab: agent sends hold action (delta=0)
            if np.random.random() < MISSED_LAB_PROB:
                action = np.array([[0.0]], dtype=np.float32)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, done_array, info = eval_env.step(action)

            true_aptt = get_true_aptt(base_env.state[0], base_env.state[1])
            aptt_hist.append(true_aptt)
            dose_hist.append(base_env.current_dose)
            if 60.0 <= true_aptt <= 80.0:
                hours_in_target += 1

            # External mortality checks
            if base_env.state[2] > STRICT_CLOT_BURDEN:
                ep_outcome = "clot"
                done = True
            elif base_env.last_lab_aptt > STRICT_BLEED_APTT:
                ep_outcome = "bleed"
                done = True
            elif done_array[0]:
                ep_outcome = "survived"
                done = True

        if ep_outcome == "survived":
            survived += 1
        elif ep_outcome == "clot":
            fatal_clots += 1
        else:
            fatal_bleeds += 1

        if len(aptt_hist) > 0:
            ttr = (hours_in_target / len(aptt_hist)) * 100.0
            metrics["ttr_list"].append(ttr)
            m = calculate_metrics(aptt_hist, dose_hist)
            for k in keys[1:]:
                metrics[k].append(m[k])

    metrics["survived"]     = survived
    metrics["clots"]        = fatal_clots
    metrics["bleeds"]       = fatal_bleeds
    metrics["total"]        = episodes
    return metrics


# ==========================================
# 5. SINGLE TRAJECTORY EXTRACTOR
# ==========================================
def get_trajectory_data(env_nom, env_rl_eval, model, seed=42):
    """
    Runs one patient through both systems with IDENTICAL starting biology.
    """
    # --- Nomogram trajectory ---
    obs, _ = env_nom.reset(seed=seed)
    nom_aptt, nom_dose = [], []
    done = False
    while not done:
        new_dose = realistic_clinical_nomogram(obs[0], env_nom.current_dose, env_nom.patient_weight)
        obs, terminated, truncated = _apply_nomogram_dose(env_nom, new_dose)
        done = terminated or truncated
        nom_aptt.append(get_true_aptt(env_nom.state[0], env_nom.state[1]))
        nom_dose.append(env_nom.current_dose)

    # Capture nomogram patient's exact biology
    saved_weight = env_nom.patient_weight
    saved_age    = env_nom.patient_age

    # --- RL trajectory (same patient) ---
    base_env = env_rl_eval.venv.envs[0]

    # Step 1: Let VecNormalize do its reset (updates running stats correctly)
    obs = env_rl_eval.reset()

    # Step 2: Override patient biology AFTER VecNormalize reset
    base_env.patient_weight = saved_weight
    base_env.patient_age    = saved_age
    base_env.state          = np.array([0.0, 100.0, 0.0, 0.0, 1.0], dtype=np.float32)
    base_env.t_ecmo         = 0
    base_env.current_dose   = 12.0
    base_env.prev_dose      = 12.0
    base_env.last_lab_aptt  = 45.0
    base_env.prev_lab_aptt  = 45.0
    base_env.error_integral = 0.0

    # Step 3: Re-normalise the initial obs so the model sees correct scaled input
    raw_obs = base_env._get_obs()
    obs = env_rl_eval.normalize_obs(raw_obs.reshape(1, -1))

    rl_aptt, rl_dose = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done_array, _ = env_rl_eval.step(action)
        done = done_array[0]
        rl_aptt.append(get_true_aptt(base_env.state[0], base_env.state[1]))
        rl_dose.append(base_env.current_dose)

    return nom_aptt, nom_dose, rl_aptt, rl_dose


# ==========================================
# 6. REPORT HELPERS
# ==========================================
def safe_mean(lst):
    return np.nanmean(lst) if len(lst) > 0 else 0.0

def print_report(nom_m, rl_m, n):
    nom_fail = 100.0 - (nom_m["survived"] / n * 100)
    rl_fail  = 100.0 - (rl_m["survived"]  / n * 100)

    print("\n" + "=" * 65)
    print(f"{'PHASE IV POPULATION TRIAL (n=' + str(n) + ')':^65}")
    print("=" * 65)
    print(f"{'Metric':<32} | {'Nomogram':^14} | {'RL Agent':^10}")
    print("-" * 65)
    rows = [
        ("Failure Rate (%)",          f"{nom_fail:.2f}%",                   f"{rl_fail:.2f}%"),
        ("  → Fatal Hemorrhages",     str(nom_m["bleeds"]),                  str(rl_m["bleeds"])),
        ("  → Fatal Thrombosis",      str(nom_m["clots"]),                   str(rl_m["clots"])),
        ("Avg TTR (%)",               f"{safe_mean(nom_m['ttr_list']):.1f}", f"{safe_mean(rl_m['ttr_list']):.1f}"),
        ("Time to Target (hrs)",      f"{safe_mean(nom_m['ttt']):.1f}",      f"{safe_mean(rl_m['ttt']):.1f}"),
        ("AUC Error (Severity)",      f"{safe_mean(nom_m['auc']):.1f}",      f"{safe_mean(rl_m['auc']):.1f}"),
        ("Pump Chattering (U/hr)",    f"{safe_mean(nom_m['chatter']):.2f}",  f"{safe_mean(rl_m['chatter']):.2f}"),
        ("Avg Pump Adjustments",      f"{safe_mean(nom_m['adjustments']):.1f}", f"{safe_mean(rl_m['adjustments']):.1f}"),
        ("Dangerous Hours (aPTT<50)", f"{safe_mean(nom_m['dangerous_hours']):.1f}", f"{safe_mean(rl_m['dangerous_hours']):.1f}"),
        ("Peak aPTT (s)",             f"{safe_mean(nom_m['peak_aptt']):.1f}",f"{safe_mean(rl_m['peak_aptt']):.1f}"),
        ("Trough aPTT (s)",           f"{safe_mean(nom_m['trough_aptt']):.1f}", f"{safe_mean(rl_m['trough_aptt']):.1f}"),
    ]
    for label, nv, rv in rows:
        print(f"  {label:<30} | {nv:^14} | {rv:^10}")
    print("=" * 65 + "\n")


# ==========================================
# 7. DASHBOARD PLOTTING
# ==========================================
def plot_dashboard(nom_m, rl_m, nom_a, nom_d, rl_a, rl_d, n, seed):
    fig = plt.figure(figsize=(22, 10))
    fig.patch.set_facecolor("#0f0f1a")

    panel_kw = dict(facecolor="#1a1a2e")
    lbl_kw   = dict(color="#ccccdd", fontsize=9)

    # ── Top: Trajectory ──────────────────────────────────────────
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=4, **panel_kw)
    ax1.set_title(
        f"Deep-Dive Case Study: Patient #{seed}  |  Apocalypse Mode  |  35% Missed Labs",
        color="white", fontsize=13, pad=10
    )
    ax1.axhspan(60, 80, color="#00ff88", alpha=0.12, label="Therapeutic Target (60-80s)")
    ax1.axhline(STRICT_BLEED_APTT, color="#ff4444", linestyle=":", linewidth=1.5,
                label=f"Fatal Bleed Threshold ({STRICT_BLEED_APTT}s)")
    ax1.axhline(50.0, color="#ffaa00", linestyle=":", linewidth=1.0,
                label="Dangerous Sub-Therapeutic (50s)")

    hours_nom = np.arange(len(nom_a)) * 4
    hours_rl  = np.arange(len(rl_a))  * 4
    ax1.plot(hours_nom, nom_a, color="#888899", linestyle="--", linewidth=2,   label="Realistic Nomogram")
    ax1.plot(hours_rl,  rl_a,  color="#4488ff", linewidth=2.5,                 label="RL Agent (SAC)")
    ax1.set_ylabel("True aPTT (seconds)", **lbl_kw)
    ax1.set_xlabel("Hours on ECMO",       **lbl_kw)
    ax1.set_xlim(0, 120); ax1.set_ylim(20, 145)
    ax1.tick_params(colors="#aaaacc")
    ax1.grid(True, alpha=0.2, color="#333355")
    leg = ax1.legend(loc="upper right", fontsize=8, framealpha=0.3)
    for t in leg.get_texts(): t.set_color("white")

    # ── Bottom panels ─────────────────────────────────────────────
    def make_bar(pos, title, labels, values, colors, ylabel, fmt=".1f", ylim=None):
        ax = plt.subplot2grid((2, 4), (1, pos), **panel_kw)
        bars = ax.bar(labels, values, color=colors, width=0.5)
        ax.set_title(title, color="white", fontsize=9)
        ax.set_ylabel(ylabel, **lbl_kw)
        ax.tick_params(colors="#aaaacc")
        ax.spines["bottom"].set_color("#333355")
        ax.spines["left"].set_color("#333355")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ylim: ax.set_ylim(*ylim)
        max_v = max(values) if max(values) > 0 else 1
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + max_v * 0.04,
                    f"{h:{fmt}}", ha="center", fontweight="bold",
                    color="white", fontsize=9)
        return ax

    nom_fail = 100.0 - (nom_m["survived"] / nom_m["total"] * 100)
    rl_fail  = 100.0 - (rl_m["survived"]  / rl_m["total"]  * 100)

    make_bar(0, f"Failure Rate (%) — n={n}",
             ["Nomogram", "RL Agent"],
             [nom_fail, rl_fail],
             ["#888899", "#44cc88"],
             "% Failed (LOWER is Better)", fmt=".2f",
             ylim=(0, max(nom_fail * 1.6, 0.5)))

    make_bar(1, f"Avg TTR (%) — n={n}",
             ["Nomogram", "RL Agent"],
             [safe_mean(nom_m["ttr_list"]), safe_mean(rl_m["ttr_list"])],
             ["#888899", "#9944ff"],
             "% Time in Target (HIGHER is Better)", ylim=(0, 100))

    make_bar(2, f"Avg AUC Error — n={n}",
             ["Nomogram", "RL Agent"],
             [safe_mean(nom_m["auc"]), safe_mean(rl_m["auc"])],
             ["#888899", "#4488ff"],
             "Error Score (LOWER is Better)")

    make_bar(3, f"Avg Dangerous Hours (aPTT<50) — n={n}",
             ["Nomogram", "RL Agent"],
             [safe_mean(nom_m["dangerous_hours"]), safe_mean(rl_m["dangerous_hours"])],
             ["#888899", "#ff6644"],
             "Hours in Danger Zone (LOWER is Better)")

    plt.tight_layout(pad=2.0)
    fname = f"v2_final_report_n{n}.png"
    plt.savefig(fname, dpi=150, facecolor=fig.get_facecolor())
    print(f"Dashboard saved → {fname}")
    plt.show()
    return fname


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    
    
    print("Loading Virtual ICU (Phase 2)...")

    raw_env  = ECMOHeparinEnv(difficulty=3)
    dummy_env = DummyVecEnv([lambda: ECMOHeparinEnv(difficulty=3)])

    try:
       
        eval_env = VecNormalize.load("models/vec_normalize.pkl", dummy_env)
        eval_env.training  = False
        eval_env.norm_reward = False
        model = SAC.load("models/best_model.zip", env=eval_env)
        print("Model and normalizer loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")
        raise
    
    N_EPISODES = 1000
    print(f"\nSimulating {N_EPISODES} patients with {int(MISSED_LAB_PROB*100)}% missed labs "
          f"and strict mortality thresholds (clot>{STRICT_CLOT_BURDEN}, "
          f"aPTT>{STRICT_BLEED_APTT}s)...\n")

    nom_m = evaluate_nomogram(raw_env,  episodes=N_EPISODES)
    rl_m  = evaluate_rl(eval_env, model, episodes=N_EPISODES)

    print_report(nom_m, rl_m, N_EPISODES)

    print("Generating trajectory case study (Patient #42)...")
    PATIENT_SEED = 42
    nom_a, nom_d, rl_a, rl_d = get_trajectory_data(raw_env, eval_env, model, seed=PATIENT_SEED)

    plot_dashboard(nom_m, rl_m, nom_a, nom_d, rl_a, rl_d, N_EPISODES, PATIENT_SEED)
