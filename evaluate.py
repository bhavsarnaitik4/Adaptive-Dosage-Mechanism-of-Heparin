import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import os

from ecmo_env import ECMOHeparinEnv

# ==========================================
# 1. THE ADVANCED HOSPITAL PROTOCOL (V2 Baseline)
# ==========================================
def advanced_nomogram_step(current_aptt, prev_aptt, current_dose, t_ecmo, renal_health):
    """A cutting-edge, multi-variable clinical titration protocol."""
    
    # Base Standard Rules
    if current_aptt < 50.0:
        adj = +2.0
    elif 50.0 <= current_aptt < 60.0:
        adj = +1.0
    elif 60.0 <= current_aptt <= 80.0:
        adj = 0.0
    elif 80.0 < current_aptt <= 100.0:
        adj = -2.0
    else:
        adj = -3.0
        
    # Modifier 4: Tubing Adsorption (Aggressive start)
    if t_ecmo < 24 and current_aptt < 50.0:
        adj = +3.0

    # Modifier 1: Trend Velocity (Dampen if shooting up too fast)
    if current_aptt >= 60.0 and (current_aptt - prev_aptt) > 15.0:
        adj -= 1.0
        
    # Modifier 2: Renal Failure (Halve the dose increases if kidneys failing)
    if renal_health < 0.5 and adj > 0:
        adj *= 0.5
        
    # Modifier 3: AT-III Resistance Ceiling (Stop pumping drug if it isn't working)
    if current_dose > 25.0 and current_aptt < 60.0:
        adj = 0.0 # DO NOT INCREASE. (Clinically: Order AT-III Transfusion)
        
    return current_dose + adj

# ==========================================
# 2. EVALUATION FUNCTION
# ==========================================
def evaluate_agent(env, model=None, is_nomogram=False, episodes=100):
    metrics = {
        "ttr_list": [],      
        "fatal_clots": 0,
        "fatal_bleeds": 0,
        "survived": 0
    }
    
    for ep in range(episodes):
        # ---> THE HEARTBEAT: Print progress every 10 patients so we know it isn't frozen!
        if (ep + 1) % 10 == 0:
            print(f"  -> Processing Patient {ep + 1}/{episodes}...")

        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            
        done = False
        hours_in_target = 0
        total_hours = 0
        
        # Save the very first lab reading to track the trend velocity
        prev_aptt = obs[0] 
        
        while not done:
            if is_nomogram:
                current_aptt = obs[0]
                current_dose = obs[1]
                
                base_env = env.envs[0] if hasattr(env, 'envs') else env
                t_ecmo = base_env.t_ecmo
                renal_health = base_env.state[4] 
                
                # Nomogram only changes dose when a new 4-hour lab arrives
                if obs[2] == 0:
                    action_dose = advanced_nomogram_step(current_aptt, prev_aptt, current_dose, t_ecmo, renal_health)
                    prev_aptt = current_aptt # Save this lab to compare against the NEXT one
                else:
                    action_dose = current_dose
                
                # Apply Action
                action_array = np.array([(action_dose / 15.0) - 1.0], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action_array)
                done = terminated or truncated
                
                if 60.0 <= base_env.last_lab_aptt <= 80.0:
                    hours_in_target += 1
                total_hours += 1
                
                if done:
                    if base_env.state[2] > base_env.CRITICAL_CLOT_BURDEN:
                        metrics["fatal_clots"] += 1
                    elif base_env.last_lab_aptt > 150.0:
                        metrics["fatal_bleeds"] += 1
                    else:
                        metrics["survived"] += 1
                        
            else:
                # RL Agent Logic
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                done = dones[0]
                
                base_env = env.envs[0] 
                if 60.0 <= base_env.last_lab_aptt <= 80.0:
                    hours_in_target += 1
                total_hours += 1
                
                if done:
                    if base_env.state[2] > base_env.CRITICAL_CLOT_BURDEN:
                        metrics["fatal_clots"] += 1
                    elif base_env.last_lab_aptt > 150.0:
                        metrics["fatal_bleeds"] += 1
                    else:
                        metrics["survived"] += 1

        metrics["ttr_list"].append((hours_in_target / total_hours) * 100)
        
    return metrics

# ==========================================
# 3. RUN EVALUATION AND PLOT
# ==========================================
if __name__ == "__main__":
    print("Loading AI Brain...")
    
    # Setup environments
    raw_env = ECMOHeparinEnv()
    
    # We must recreate the exact FrameStack wrapper used in training
    vec_env = DummyVecEnv([lambda: ECMOHeparinEnv()])
    stacked_env = VecFrameStack(vec_env, n_stack=4)
    
    try:
        model = SAC.load("models/sac_heparin_brain")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    N_EPISODES = 100
    
    print(f"Simulating {N_EPISODES} patients using Advance Hospital Nomogram...")
    nomogram_metrics = evaluate_agent(raw_env, is_nomogram=True, episodes=N_EPISODES)
    
    print(f"Simulating {N_EPISODES} patients using Trained SAC Agent...")
    rl_metrics = evaluate_agent(stacked_env, model=model, is_nomogram=False, episodes=N_EPISODES)
    
    # --- PLOTTING ---
    print("\nGenerating final report graphs...")
    
    # Calculate Averages
    nom_avg_ttr = np.mean(nomogram_metrics["ttr_list"])
    rl_avg_ttr = np.mean(rl_metrics["ttr_list"])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Graph 1: Time in Therapeutic Range
    labels = ['Hospital Nomogram', 'RL Agent (SAC)']
    ttr_values = [nom_avg_ttr, rl_avg_ttr]
    axes[0].bar(labels, ttr_values, color=['gray', 'blue'])
    axes[0].set_ylabel('% Time in Therapeutic Range (TTR)')
    axes[0].set_title('Dosage Accuracy (Higher is Better)')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(ttr_values):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

    # Graph 2: Catastrophes
    x = np.arange(2)
    width = 0.35
    
    nom_events = [nomogram_metrics["fatal_clots"], nomogram_metrics["fatal_bleeds"]]
    rl_events = [rl_metrics["fatal_clots"], rl_metrics["fatal_bleeds"]]
    
    axes[1].bar(x - width/2, nom_events, width, label='Nomogram', color='gray')
    axes[1].bar(x + width/2, rl_events, width, label='RL Agent', color='red')
    
    axes[1].set_ylabel('Number of Patients')
    axes[1].set_title(f'Catastrophic Events out of {N_EPISODES} Patients')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Fatal Clots (Machine Failure)', 'Fatal Bleeds (Hemorrhage)'])
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('final_coursework_results.png')
    plt.show()
    
    print("\n--- FINAL RESULTS ---")
    print(f"Nomogram Survival Rate: {nomogram_metrics['survived']}/{N_EPISODES}")
    print(f"RL Agent Survival Rate: {rl_metrics['survived']}/{N_EPISODES}")
    print("\nPlot saved as 'final_coursework_results.png'. You are ready to write the report!")