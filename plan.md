# 48-Hour Execution Plan: ECMO-Heparin RL (V2.5)

## 🛠️ Phase 0: Environment Setup (Hour 0 - 1)

* [ ] **0.1 Create Virtual Environment:** Set up a clean Python 3.10+ environment.
* [ ] **0.2 Install Core Libraries:** Run `pip install gymnasium stable-baselines3[extra] scipy numpy pandas matplotlib seaborn`.
* [ ] **0.3 File Structure:** Create the following blank files:
* `physiology.py` (For the ODE math)
* `ecmo_env.py` (For the Gymnasium environment)
* `train.py` (For SB3 SAC training)
* `baseline.py` (For the nomogram logic)
* `evaluate.py` (For generating plots)



---

## 🧬 Phase 1: The Mechanistic Engine (Day 1 - Morning)

*File: `physiology.py*`

* [ ] **1.1 Define Constants:** Create a dictionary or dataclass for your PK/PD constants (e.g., $V_{max}, K_m, k_{renal}, k_{ecmo}, k_{consume}$). *Use plausible dummy values to start; you can tune them later.*
* [ ] **1.2 Code the Step Function:** Write a function `step_physiology(state, dose, dt=1)` that takes the current state `[C, AT, Omega]` and outputs the next state using simple Euler integration.
* *Formula 1:* Update Heparin Concentration ($C$).
* *Formula 2:* Update AT-III ($AT$) with shear and consumption.
* *Formula 3:* Update Clot Burden ($\Omega$).


* [ ] **1.3 Code the Observation Function:** Write a function `get_true_aptt(C, AT)` using the $E_{max}$ equation.
* [ ] **1.4 Test Block 1:** Write a quick script to loop `step_physiology` 120 times (hours) with a constant dose. Plot the `true_aptt` to ensure the math doesn't explode to infinity or drop to negative numbers.

---

## 🌍 Phase 2: The Gymnasium Environment (Day 1 - Afternoon)

*File: `ecmo_env.py*`

* [ ] **2.1 Class Setup:** Create `class ECMOHeparinEnv(gymnasium.Env):`
* [ ] **2.2 Define Spaces:**
* `self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))`
* `self.observation_space = spaces.Box(low=0, high=200, shape=(3,))` *(Vector: `[o_aPTT, prev_dose, time_since_lab]`)*


* [ ] **2.3 Code `reset()`:**
* Randomize patient baseline weight ($75 \pm 15$ kg) and baseline AT-III ($80 \pm 10$%).
* Reset hidden state: `t = 0`, `C = 0`, `Omega = 0`.
* Return initial observation vector and `{}` info dict.


* [ ] **2.4 Code `step(action)` - Guardrails:**
* Map action `[-1, 1]` to clinical dose `[0, 30]`.
* Implement Action Masking: If `prev_aPTT > 120`, override dose to `0`. If `|dose - prev_dose| > 5`, clip the dose.


* [ ] **2.5 Code `step(action)` - POMDP Logic:**
* Advance the internal clock by 1 hour.
* Call `step_physiology()` to update hidden states.
* **Sparse Logic:** If `t % 4 == 0`, update `o_aPTT` with noise. Else, `time_since_lab += 1`.


* [ ] **2.6 Code `step(action)` - Rewards & Terminals:**
* Calculate continuous reward (TTR + Smoothness).
* Check terminal conditions (Clot > limit, aPTT > 150, Time >= 120).
* Return `obs, reward, terminated, truncated, info`.


* [ ] **2.7 Test Block 2:** Use `stable_baselines3.common.env_checker.check_env` to verify your environment complies with the Gym API.

---

## 🧠 Phase 3: The RL Architecture Hack (Day 1 - Evening)

*File: `train.py*`

* [ ] **3.1 Frame Stacking:** Import `DummyVecEnv` and `VecFrameStack` from `stable_baselines3.common.vec_env`.
* [ ] **3.2 Wrap Environment:** Wrap your `ECMOHeparinEnv` in the vectorizer, then apply `VecFrameStack(env, n_stack=4)`. *This solves your POMDP problem without needing custom RNNs.*
* [ ] **3.3 Initialize SAC:** Instantiate the SAC model from SB3.
* `model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_tensorboard/")`


* [ ] **3.4 The Overnight Run:** Set it to train for ~300,000 to 500,000 timesteps (`model.learn(total_timesteps=500000)`). Let it run overnight. Save the model at the end (`model.save("sac_heparin_v2_5")`).

---

## 🏥 Phase 4: The Baseline Nomogram (Day 2 - Morning)

*File: `baseline.py*`

* [ ] **4.1 Code the Heuristic:** Write a Python function `nomogram_step(current_aptt, current_dose)` that mimics standard hospital protocol (if <50 add 2, if >80 drop 2, etc.).
* [ ] **4.2 Create the Simulation Loop:** Write a script that runs a fresh environment using *only* the `nomogram_step` to decide actions.
* [ ] **4.3 Test Block 4:** Run 10 episodes with the nomogram. Print the outputs to ensure it correctly adjusts the dose when the labs go out of range.

---

## 📊 Phase 5: Evaluation & Plotting (Day 2 - Afternoon)

*File: `evaluate.py*`

* [ ] **5.1 Bulk Evaluation:** Write a script to run 1,000 simulated episodes using the trained SAC model, and 1,000 using the Baseline Nomogram.
* [ ] **5.2 Calculate Metrics:** For both groups, calculate:
* Average % Time in Therapeutic Range (TTR).
* Total count of Catastrophic Bleeds (Terminated via aPTT > 150).
* Total count of Catastrophic Clots (Terminated via $\Omega > \Omega_{limit}$).


* [ ] **5.3 Generate Plot 1 (Bar Chart):** Use `matplotlib`/`seaborn` to plot the TTR, Bleeds, and Clots side-by-side (SAC vs. Baseline).
* [ ] **5.4 Generate Plot 2 (Trajectory):** Run *one* episode with SAC and save every variable at every timestep.
* Plot a multi-axis graph:
* Top subplot: True aPTT (line) + Sparse Noisy Labs (scatter dots). Shade the 60-80 "safe zone" in green.
* Middle subplot: True AT-III level dropping over time.
* Bottom subplot: Agent's Dose (bar chart). *Highlight how the agent increases the dose as AT-III drops!*




* [ ] **5.5 Save Assets:** Save these images to an `assets/` folder for your coursework report.

---

## 📝 Phase 6: Report Writing (Day 2 - Evening)

* [ ] **6.1 Introduction:** Frame the problem (ECMO Heparin is highly non-linear, standard protocols fail).
* [ ] **6.2 Methodology:** Explain the POMDP formulation. Clearly state why you used Frame Stacking to handle sparse, 4-hour delayed labs. Show your ODE equations.
* [ ] **6.3 Results:** Paste your bar charts and trajectory plots.
* [ ] **6.4 Future Work (The "V3" ideas):** Briefly write 2 paragraphs explaining that future iterations would include dual-action AT-III control, Ensemble Critics for uncertainty estimation, and Bayesian calibration against the MIMIC-IV dataset. *(This is where you earn the A+ by showing you know the limitations of your 48-hour build).*