import gymnasium as gym
import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from ecmo_env import ECMOHeparinEnv

def make_env(difficulty=3, rank=0):
    def _init():
        env = ECMOHeparinEnv(difficulty=difficulty)
        env.reset(seed=rank)
        return env
    return _init

if __name__ == "__main__":
    print("Initializing the V2 Training Run...")
    os.makedirs("./models/best/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)
    
    NUM_CORES = 8
    raw_env = SubprocVecEnv([make_env(difficulty=3, rank=i) for i in range(NUM_CORES)])
    env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    eval_raw_env = DummyVecEnv([lambda: ECMOHeparinEnv(difficulty=3)])
    eval_env = VecNormalize(eval_raw_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    print(f"Building SAC Agent. Cores: {NUM_CORES} | Target: 2,000,000 Steps")
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256, 256], log_std_init=-2.0),
        learning_rate=3e-4,          # AGGRESSIVE: Faster weight updates
        batch_size=256,              # FASTER: More updates per epoch
        buffer_size=300_000,         # TIGHTER: Focus on recent, good data
        learning_starts=10_000,      
        tau=0.005,                   
        gamma=0.99,                 
        tensorboard_log="./logs/"
    )
    
    TRAIN_STEPS = 1_000_000 
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/best/',
        log_path='./logs/',
        eval_freq=int(50_000 / NUM_CORES), 
        n_eval_episodes=20,
        deterministic=True
    )
    
    model.learn(total_timesteps=TRAIN_STEPS, callback=eval_callback, progress_bar=True)
    
    env.save("models/best/vec_normalize.pkl") 
    print("Training Complete. Models saved in /models/best/.")