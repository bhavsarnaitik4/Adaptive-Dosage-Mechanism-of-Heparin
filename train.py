import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import os

# Import your custom environment
from ecmo_env import ECMOHeparinEnv

if __name__ == "__main__":
    print("Initializing the ECMO Matrix...")

    # 1. Create the Environment
    env = ECMOHeparinEnv()

    # 2. The POMDP Hack: Frame Stacking
    # We wrap the environment in a vectorizer, then stack the last 4 observations.
    # This gives the AI "memory" of the 4-hour gap between blood labs.
    vec_env = DummyVecEnv([lambda: env])
    stacked_env = VecFrameStack(vec_env, n_stack=4)

    # 3. Build the Soft Actor-Critic (SAC) Agent
    print("Building the SAC Agent...")
    model = SAC(
        "MlpPolicy", 
        stacked_env, 
        verbose=1, 
        learning_rate=0.0003,
        tensorboard_log="./sac_tensorboard/"
    )

    # 4. Train the Agent!
    # For a coursework project, 150,000 to 300,000 timesteps is usually enough 
    # to see it learn to avoid killing the patient.
    TRAIN_STEPS = 150000 
    
    print(f"Starting Training for {TRAIN_STEPS} hours of simulated ICU time...")
    model.learn(total_timesteps=TRAIN_STEPS, progress_bar=True)

    # 5. Save the trained "Brain"
    os.makedirs("models", exist_ok=True)
    model.save("models/sac_heparin_brain")
    print("\nTraining Complete! AI Brain saved to models/sac_heparin_brain.zip")