#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback, CallbackList

# Try to import XPlaneGym environment
try:
    import XPlaneGym
except ImportError:
    raise ImportError("Please make sure XPlaneGym package is installed")

def make_env(env_id, rank, seed=0):
    """
    Helper function to create environments
    """
    def _init():
        # Use the new Gymnasium API to set random seed
        env = gym.make(env_id, continuous_actions=true)
        env = Monitor(env)
        # Set random seed when creating environment, instead of calling env.seed()
        return env
    return _init

def train_ppo(
    env_id="XPlane-v0",
    total_timesteps=1000000,
    save_path="./models",
    log_path="./logs",
    save_freq=10000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    resume=False,
    checkpoint_path=None
):
    """
    Train XPlaneGym environment using PPO
    
    Parameters:
        env_id: Environment ID
        total_timesteps: Total training steps
        save_path: Model save path
        log_path: Log save path
        save_freq: Save frequency (steps)
        learning_rate: Learning rate
        n_steps: Number of steps to collect before each update
        batch_size: Batch size
    """
    # Create output directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Create vectorized environment
    env = DummyVecEnv([make_env(env_id, i) for i in range(1)])
    
    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // env.num_envs,
        save_path=save_path,
        name_prefix="ppo_model"
    )

    # Real-time console logger callback
    class ConsoleLoggerCallback(BaseCallback):
        """
        在 episode 结束时，将奖励与长度打印到控制台，便于实时观察训练效果。
        依赖 Monitor 包装器在 episode 结束时在 info 中注入 'episode' 信息。
        """
        def __init__(self, verbose: int = 0):
            super().__init__(verbose)

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                episode_info = info.get("episode")
                if episode_info is not None:
                    reward = episode_info.get("r")
                    length = episode_info.get("l")
                    time_elapsed = episode_info.get("t", None)
                    if time_elapsed is not None:
                        print(f"[Step {self.num_timesteps}] episode_reward={reward:.2f}, length={length}, time={time_elapsed:.2f}s")
                    else:
                        print(f"[Step {self.num_timesteps}] episode_reward={reward:.2f}, length={length}")
            return True

    callback_list = CallbackList([checkpoint_callback, ConsoleLoggerCallback()])

    # -------- 断点续训：尝试从 checkpoint 和 VecNormalize 统计恢复 --------
    loaded_from_checkpoint = False
    model = None

    if resume:
        # 1) 选择要加载的 checkpoint
        ckpt_to_load = None
        if checkpoint_path and os.path.isfile(checkpoint_path):
            ckpt_to_load = checkpoint_path
        else:
            candidates = glob.glob(os.path.join(save_path, "ppo_model*_steps.zip"))
            if candidates:
                candidates.sort(key=lambda p: os.path.getmtime(p))
                ckpt_to_load = candidates[-1]

        # 2) 加载 VecNormalize 统计
        stats_path = os.path.join(save_path, "vec_normalize.pkl")
        if os.path.exists(stats_path):
            try:
                env = VecNormalize.load(stats_path, env)
                env.training = True
                env.norm_reward = True
                env.clip_obs = 10.
                print(f"Loaded VecNormalize stats from: {stats_path}")
            except Exception as e:
                print(f"Warning: failed to load VecNormalize stats: {e}")

        # 3) 加载模型
        if ckpt_to_load:
            try:
                print(f"Resuming from checkpoint: {ckpt_to_load}")
                model = PPO.load(ckpt_to_load, env=env, tensorboard_log=log_path)
                loaded_from_checkpoint = True
            except Exception as e:
                print(f"Warning: failed to load checkpoint {ckpt_to_load}: {e}")

    # 若未能加载，则新建模型
    if model is None:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_path,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size
        )

    # 计算已训练步数与剩余步数
    already_trained_steps = int(getattr(model, "num_timesteps", 0) or 0)
    remaining_steps = int(max(0, total_timesteps - already_trained_steps)) if resume else int(total_timesteps)

    if remaining_steps == 0:
        print(f"Target total {total_timesteps} already reached (trained: {already_trained_steps}). Nothing to do.")
        model.save(os.path.join(save_path, "ppo_final_model"))
        env.save(os.path.join(save_path, "vec_normalize.pkl"))
        return model

    print(
        f"Starting PPO training: target_total={total_timesteps}, already_trained={already_trained_steps}, remaining={remaining_steps}"
    )
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=callback_list,
            tb_log_name="ppo_run",
            log_interval=1,
            progress_bar=True,
            reset_num_timesteps=not loaded_from_checkpoint
        )
    except TypeError:
        # 兼容较老版本的 stable-baselines3（无 progress_bar 参数）
        model.learn(
            total_timesteps=remaining_steps,
            callback=callback_list,
            tb_log_name="ppo_run",
            log_interval=1,
            reset_num_timesteps=not loaded_from_checkpoint
        )
    
    # Save final model and normalization parameters
    model.save(os.path.join(save_path, "ppo_final_model"))
    env.save(os.path.join(save_path, "vec_normalize.pkl"))
    
    print(f"Training completed! Time taken: {time.time() - start_time:.2f} seconds")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on XPlaneGym with optional resume.")
    parser.add_argument("--env_id", type=str, default="XPlane-v0")
    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--save_path", type=str, default="./models")
    parser.add_argument("--log_path", type=str, default="./logs")
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--resume", type=str, default="false", help="true/false to resume from latest checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="specific checkpoint .zip to load")

    args = parser.parse_args()

    resume_flag = str(args.resume).lower() in ["1", "true", "t", "yes", "y"]

    train_ppo(
        env_id=args.env_id,
        total_timesteps=args.total_timesteps,
        save_path=args.save_path,
        log_path=args.log_path,
        save_freq=args.save_freq,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        resume=resume_flag,
        checkpoint_path=args.checkpoint_path
    )
