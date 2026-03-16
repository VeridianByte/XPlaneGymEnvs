#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import glob
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

# Import for custom callbacks
from custom_callbacks import get_callbacks

# Try to import XPlaneGym environment
try:
    import XPlaneGym
except ImportError:
    raise ImportError("Please ensure XPlaneGym package is installed")

def make_env(env_id, rank=0):
    """
    Helper function to create environment
    """
    def _init():
        env = gym.make(env_id, continuous_actions=False)
        env = Monitor(env)
        return env
    return _init

def train_dqn(
    env_id="XPlane-v0",
    total_timesteps=500000,
    save_path="./models",
    log_path="./logs",
    save_freq=10000,
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    tau=1.0,
    target_update_interval=1000,
    train_freq=4,
    gradient_steps=1,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    use_custom_callbacks=True,
    resume=False,
    checkpoint_path=None
):
    """
    Train XPlaneGym environment using DQN
    
    Parameters:
        env_id: Environment ID
        total_timesteps: Total training steps
        save_path: Model save path
        log_path: Log save path
        save_freq: Save frequency (steps)
        learning_rate: Learning rate
        buffer_size: Experience replay buffer size
        learning_starts: Steps before learning starts
        batch_size: Batch size
        gamma: Discount factor
        tau: Target network soft update coefficient
        target_update_interval: Target network update interval
        train_freq: Training frequency
        gradient_steps: Gradient steps per update
        exploration_fraction: Exploration steps as fraction of total steps
        exploration_initial_eps: Initial exploration rate
        exploration_final_eps: Final exploration rate
        use_custom_callbacks: Whether to use custom callbacks
    """
    # Create output directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Create vectorized environment
    env = DummyVecEnv([make_env(env_id)])

    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Set up callbacks
    # Real-time console logger callback
    class ConsoleLoggerCallback(BaseCallback):
        """
        在 episode 结束时打印奖励与长度。
        依赖 Monitor 在 info 中注入 'episode' 字段。
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

    # Base checkpoint callback
    base_ckpt_cb = CheckpointCallback(
        save_freq=save_freq // (env.num_envs if hasattr(env, 'num_envs') else 1),
        save_path=save_path,
        name_prefix="dqn_model"
    )

    # Compose callbacks
    callbacks = CallbackList([base_ckpt_cb, ConsoleLoggerCallback()])
    
    # -------- 断点续训：尝试从 checkpoint 与 VecNormalize 恢复 --------
    loaded_from_checkpoint = False
    model = None

    if resume:
        # 1) 选择要加载的 checkpoint
        ckpt_to_load = None
        if checkpoint_path and os.path.isfile(checkpoint_path):
            ckpt_to_load = checkpoint_path
        else:
            candidates = glob.glob(os.path.join(save_path, "dqn_model*_steps.zip"))
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
                model = DQN.load(ckpt_to_load, env=env, tensorboard_log=log_path)
                loaded_from_checkpoint = True
            except Exception as e:
                print(f"Warning: failed to load checkpoint {ckpt_to_load}: {e}")

    # 若未能加载，则新建模型
    if model is None:
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_path,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            target_update_interval=target_update_interval,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps
        )
    
    # 计算已训练步数与剩余步数
    already_trained_steps = int(getattr(model, "num_timesteps", 0) or 0)
    remaining_steps = int(max(0, total_timesteps - already_trained_steps)) if resume else int(total_timesteps)

    if remaining_steps == 0:
        print(f"Target total {total_timesteps} already reached (trained: {already_trained_steps}). Nothing to do.")
        model.save(os.path.join(save_path, "dqn_final_model"))
        env.save(os.path.join(save_path, "vec_normalize.pkl"))
        return model

    print(f"Starting DQN training: target_total={total_timesteps}, already_trained={already_trained_steps}, remaining={remaining_steps}")
    print(f"Exploration settings: initial rate={exploration_initial_eps}, final rate={exploration_final_eps}, fraction={exploration_fraction}")
    print(f"Optimization settings: learning rate={learning_rate}, batch size={batch_size}, buffer size={buffer_size}")
    
    start_time = time.time()
    
    # Train model
    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=callbacks,
            tb_log_name="dqn_run",
            log_interval=1,
            progress_bar=True,
            reset_num_timesteps=not loaded_from_checkpoint
        )
    except TypeError:
        # 兼容较老版本（无 progress_bar 参数）
        model.learn(
            total_timesteps=remaining_steps,
            callback=callbacks,
            tb_log_name="dqn_run",
            log_interval=1,
            reset_num_timesteps=not loaded_from_checkpoint
        )
    
    # Save final model and VecNormalize stats
    model.save(os.path.join(save_path, "dqn_final_model"))
    env.save(os.path.join(save_path, "vec_normalize.pkl"))
    
    print(f"Training complete! Time taken: {time.time() - start_time:.2f} seconds")
    
    # Evaluate final model performance
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Final model evaluation: Average reward = {mean_reward:.2f} ± {std_reward:.2f}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on XPlaneGym with optional resume.")
    parser.add_argument("--env_id", type=str, default="XPlane-v0")
    parser.add_argument("--total_timesteps", type=int, default=50000)
    parser.add_argument("--save_path", type=str, default="./models")
    parser.add_argument("--log_path", type=str, default="./logs")
    parser.add_argument("--save_freq", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--target_update_interval", type=int, default=1000)
    parser.add_argument("--train_freq", type=int, default=4)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--exploration_fraction", type=float, default=0.1)
    parser.add_argument("--exploration_initial_eps", type=float, default=1.0)
    parser.add_argument("--exploration_final_eps", type=float, default=0.05)
    parser.add_argument("--use_custom_callbacks", type=str, default="true")
    parser.add_argument("--resume", type=str, default="false")
    parser.add_argument("--checkpoint_path", type=str, default=None)

    args = parser.parse_args()

    use_cbs = str(args.use_custom_callbacks).lower() in ["1", "true", "t", "yes", "y"]
    resume_flag = str(args.resume).lower() in ["1", "true", "t", "yes", "y"]

    train_dqn(
        env_id=args.env_id,
        total_timesteps=args.total_timesteps,
        save_path=args.save_path,
        log_path=args.log_path,
        save_freq=args.save_freq,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        target_update_interval=args.target_update_interval,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        use_custom_callbacks=use_cbs,
        resume=resume_flag,
        checkpoint_path=args.checkpoint_path
    )
