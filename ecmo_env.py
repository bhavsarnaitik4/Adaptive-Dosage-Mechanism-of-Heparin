import gymnasium as gym
from gymnasium import spaces
import numpy as np
from physiology import step_physiology, get_true_aptt

class ECMOHeparinEnv(gym.Env):
    def __init__(self, difficulty=3):
        super(ECMOHeparinEnv, self).__init__()
        self.difficulty = difficulty
        
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -100.0, -10.0, 0.0, 0.0]), 
            high=np.array([300.0, 300.0, 40.0, 1.0, 1.0, 2.0, 100.0, 10.0, 3.0, 3.0]), 
            dtype=np.float32
        )
        
        self.MAX_HOURS = 120
        self.CRITICAL_CLOT_BURDEN = 15.0 
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.patient_weight = np.random.uniform(50.0, 120.0)
        self.patient_age = np.random.uniform(18.0, 85.0)
        
        # [C, AT-III, Clot, Fluid_Overload, Renal_Health]
        self.state = np.array([0.0, 100.0, 0.0, 0.0, 1.0], dtype=np.float32)
        
        if self.difficulty == 1:
            self.state[3] = 0.0 
            self.state[4] = 1.0 
            
        self.t_ecmo = 0
        self.current_dose = 12.0
        self.prev_dose = 12.0
        self.last_lab_aptt = 45.0
        self.prev_lab_aptt = 45.0
        self.error_integral = 0.0
        
        return self._get_obs(), {}

    def _get_obs(self):
        aptt_velocity = self.last_lab_aptt - self.prev_lab_aptt
        return np.array([
            self.last_lab_aptt, 
            self.prev_lab_aptt,
            self.current_dose, 
            self.t_ecmo / 120.0,           # The Clock
            self.state[4],                 # eGFR / Renal Health
            self.state[3] / 10.0,          # Fluid Balance (Normalized)
            aptt_velocity, 
            self.error_integral / 100.0,   
            self.patient_weight / 85.0,  
            self.patient_age / 50.0      
        ], dtype=np.float32)

    def step(self, action):
        # 1. DECODE NORMALIZED ACTION INTO ASYMMETRIC DELTA
        max_increase = 3.0
        max_decrease = 6.0
        delta = action[0] * (max_increase if action[0] > 0 else max_decrease)
        
        self.prev_dose = self.current_dose
        self.current_dose = np.clip(self.current_dose + delta, 5.0, 35.0)

        # 2. 4-HOUR INTERNAL BIOLOGY LOOP
        for hr in range(4):
            self.state = step_physiology(self.state, self.current_dose, self.t_ecmo, self.patient_weight, self.patient_age)
            if self.difficulty == 1:
                self.state[3] = 0.0 
                self.state[4] = 1.0 
            self.t_ecmo += 1
            
            hourly_aptt = get_true_aptt(self.state[0], self.state[1])
            self.error_integral = (self.error_integral * 0.95) + (hourly_aptt - 70.0)

        # 3. GET LAB RESULT
        true_aptt = get_true_aptt(self.state[0], self.state[1])
        self.prev_lab_aptt = self.last_lab_aptt
        
        if self.difficulty == 3 and np.random.random() < 0.10:
            pass # Missed lab
        else:
            self.last_lab_aptt = true_aptt + np.random.normal(0, 2.0)
            
    
        error = true_aptt - 70.0
        
    
        target_penalty = 0.06 * (error ** 2) 
        
    
        action_penalty = 8.0 * (abs(delta) ** 1.5)
        
        
        if 60.0 <= true_aptt <= 80.0:
            centering_bonus = 2.0 * (1.0 - abs(true_aptt - 70.0) / 10.0)
        else:
            centering_bonus = 0.0
            
        reward = centering_bonus - target_penalty - action_penalty 
        
        # 5. TERMINAL CONDITIONS
        terminated = False
        if self.state[2] > self.CRITICAL_CLOT_BURDEN or self.last_lab_aptt > 150.0:
            terminated = True
            reward -= 2000.0
        elif self.state[2] > 5.0:  
            reward -= 50.0
        elif self.last_lab_aptt > 120.0: 
            reward -= 30.0
            
        truncated = bool(self.t_ecmo >= self.MAX_HOURS)
        if truncated and not terminated: reward += 50.0
            
        return self._get_obs(), float(reward), terminated, truncated, {}