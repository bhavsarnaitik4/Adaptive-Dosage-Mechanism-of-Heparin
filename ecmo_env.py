import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Import the math engine we built in Phase 1
from physiology import step_physiology, get_true_aptt

class ECMOHeparinEnv(gym.Env):
    """
    Simulated ICU Environment for Continuous Heparin Dosing on ECMO.
    Formulated as a Partially Observable Markov Decision Process (POMDP).
    """
    
    def __init__(self):
        super(ECMOHeparinEnv, self).__init__()
        
        # ---------------------------------------------------------
        # 1. ACTION SPACE (The IV Pump)
        # ---------------------------------------------------------
        # The AI outputs a continuous number between -1 and 1. 
        # We will mathematically stretch this to 0 to 30 Units/kg/hr.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # ---------------------------------------------------------
        # 2. OBSERVATION SPACE (What the AI sees)
        # ---------------------------------------------------------
        # The AI only sees 3 things: [Reported aPTT Lab, Previous Dose, Hours Since Last Lab]
        # Max plausible values are [250 seconds, 30 Units/kg/hr, 4 hours]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32), 
            high=np.array([250.0, 30.0, 4.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        # Internal State tracking
        self.t_ecmo = 0
        self.state = [0.0, 100.0, 0.0] # C, AT-III, Clot Burden
        self.last_lab_aptt = 30.0
        self.time_since_lab = 0
        self.prev_dose = 18.0          # Standard starting protocol
        
        # Safety Limits
        self.MAX_HOURS = 120
        self.CRITICAL_CLOT_BURDEN = 10.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t_ecmo = 0
        
        # DOMAIN RANDOMIZATION (The ultimate AI stress test)
        self.patient_weight = np.random.uniform(50.0, 100.0) 
        starting_at3 = np.random.uniform(80.0, 100.0)
        
        # Does the patient start with healthy kidneys? (1.0 = perfect, 0.5 = already failing)
        starting_renal_health = np.random.uniform(0.5, 1.0)
        
        # State: [C, AT-III, Clot, Fluid_Overload, Renal_Health]
        self.state = [0.0, starting_at3, 0.0, 0.0, starting_renal_health] 
        
        self.prev_dose = 18.0
        self.last_lab_aptt = get_true_aptt(self.state[0], self.state[1])
        self.time_since_lab = 0
        
        obs = np.array([self.last_lab_aptt, self.prev_dose, self.time_since_lab], dtype=np.float32)
        return obs, {}

    def step(self, action):
        """Advances the simulation by 1 hour based on the AI's chosen dose"""
        
        # 1. Action Translation: Stretch [-1, 1] to [0, 30] U/kg/hr
        raw_dose = (action[0] + 1.0) * 15.0
        
        # ---------------------------------------------------------
        # 2. CLINICAL GUARDRAILS (Action Masking)
        # ---------------------------------------------------------
        # Rule A: Smoothness. A doctor wouldn't wildly spin the IV dial. Clip change to max 5 U/kg/hr.
        dose = np.clip(raw_dose, self.prev_dose - 5.0, self.prev_dose + 5.0)
        
        # Rule B: The Hold Protocol. If the last lab showed massive bleeding, force the pump OFF.
        if self.last_lab_aptt > 120.0:
            dose = 0.0 
            
        # ---------------------------------------------------------
        # 3. BIOLOGICAL UPDATE (The Math Engine)
        # ---------------------------------------------------------
        # Pass the randomized weight into the simulator
        self.state = step_physiology(self.state, dose, self.t_ecmo, self.patient_weight, dt=1.0)
        self.t_ecmo += 1
        self.prev_dose = dose
        
        # Get true blood thickness (Hidden from AI)
        true_aptt = get_true_aptt(self.state[0], self.state[1])
        
        # ---------------------------------------------------------
        # 4. POMDP LOGIC: THE BLINDFOLD (Sparse Delayed Labs)
        # ---------------------------------------------------------
        # Labs are only drawn every 4 hours. 
        if self.t_ecmo % 4 == 0:
            # Add a slight measurement error (Gaussian noise) to simulate a real lab machine
            noise = np.random.normal(0, 2.0)
            self.last_lab_aptt = true_aptt + noise
            self.time_since_lab = 0
        else:
            self.time_since_lab += 1
            
        obs = np.array([self.last_lab_aptt, self.prev_dose, self.time_since_lab], dtype=np.float32)
        
        # ---------------------------------------------------------
        # 5. REWARD FUNCTION (The Scoreboard)
        # ---------------------------------------------------------
        reward = 0.0
        terminated = False
        
        # Continuous Reward: +1 if the TRUE aPTT is safely in the green zone
        if 60.0 <= true_aptt <= 80.0:
            reward += 1.0
            
        # Continuous Penalty: Slightly penalize jittery, unstable IV pump changes
        reward -= 0.1 * abs(dose - self.prev_dose)
        
        # Terminal Catastrophes: Did the patient die?
        clot_burden = self.state[2]
        
        if clot_burden > self.CRITICAL_CLOT_BURDEN:
            reward -= 100.0  # Massive penalty for fatal clot
            terminated = True
            
        elif true_aptt > 150.0:
            reward -= 100.0  # Massive penalty for fatal brain bleed
            terminated = True
            
        elif self.t_ecmo >= self.MAX_HOURS:
            reward += 50.0   # Massive reward for keeping them alive for 5 days!
            terminated = True

        return obs, float(reward), terminated, False, {}

# ==========================================
# TEST BLOCK (Validates the Environment)
# ==========================================
if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    
    env = ECMOHeparinEnv()
    
    # 1. Run the official Gymnasium API compliance checker
    print("Checking Environment Compliance...")
    check_env(env, warn=True)
    print("Environment passes Gymnasium standards!\n")
    
    # 2. Run a dummy simulation with random AI actions
    obs, info = env.reset()
    print(f"Starting Observation: aPTT={obs[0]:.1f}, PrevDose={obs[1]}, HoursSinceLab={obs[2]}")
    
    for i in range(12):
        # AI picks a random action between -1 and 1
        action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Hour {i+1} | Obs: aPTT={obs[0]:.1f}, PrevDose={obs[1]:.1f}, HrsOld={obs[2]:.0f} | Reward: {reward:.1f}")
        
        if terminated:
            print("Episode Terminated Early.")
            break