import os
import sys
import numpy as np
import cv2

import stable_baselines3 as sb3
import tactile_gym.rl_envs # register env
from datetime import datetime
from sb3_contrib import RAD_PPO, RAD_SAC
from tactile_gym.sb3_helpers.custom_policies import MViTacRL_SAC, MViTacRL_PPO
from tactile_gym.sb3_helpers.rl_utils import make_eval_env
from tactile_gym.utils.general_utils import load_json_obj

seed = 0

def collect_data_and_eval(
    model, env, n_eval_episodes=10, deterministic=True
):
    # Create data collection directories
    base_data_dir = os.path.join("vtgen/data", datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    episode_rewards, episode_lengths = [], []
    
    for episode in range(n_eval_episodes):
        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        # obj_name = obs['object_name']

        # Create episode-specific directories
        episode_dir = os.path.join(base_data_dir, f"episode_{episode:06d}")
        # episode_dir = os.path.join(base_data_dir, obj_name, f"episode_{episode:06d}")
        visual_dir = os.path.join(episode_dir, "visual")
        tactile_dir = os.path.join(episode_dir, "tactile")
        
        # Create directories if they don't exist
        os.makedirs(visual_dir, exist_ok=True)
        os.makedirs(tactile_dir, exist_ok=True)

        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, _info = env.step(action)
            episode_reward += reward

            # Get combined visual and tactile observation (512x1024)
            combined_obs = env.render(mode="rgb_array")
            
            # Split the observation into visual (left) and tactile (right)
            height, width = combined_obs.shape[:2]
            split_point = width // 2
            
            visual_obs = combined_obs[:, :split_point]  # Left half (512x512)
            tactile_obs = combined_obs[:, split_point:]  # Right half (512x512)
            
            # Save observations as images
            frame_id = f"{episode_length:06d}.png"
            episode_length += 1
            
            # Save visual observation
            visual_path = os.path.join(visual_dir, frame_id)
            cv2.imwrite(visual_path, cv2.cvtColor(visual_obs, cv2.COLOR_BGR2RGB))
            
            # Save tactile observation
            tactile_path = os.path.join(tactile_dir, frame_id)
            cv2.imwrite(tactile_path, cv2.cvtColor(tactile_obs, cv2.COLOR_BGR2RGB))

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    return episode_rewards, episode_lengths

def final_evaluation(
    saved_model_dir,
    n_eval_episodes,
    seed=None,
    deterministic=True,
    show_gui=True,
    show_tactile=True,
):
    rl_params = load_json_obj(os.path.join(saved_model_dir, "rl_params"))
    algo_params = load_json_obj(os.path.join(saved_model_dir, "algo_params"))
    rl_params["image_size"] = [512, 512]

    # Create the evaluation env
    eval_env = make_eval_env(
        rl_params["env_name"],
        rl_params,
        show_gui=show_gui,
        show_tactile=show_tactile,
    )

    # Load the trained model
    model_path = os.path.join(saved_model_dir, "trained_models", "best_model")

    # Create the model with hyper params
    if rl_params["algo_name"] == "ppo":
        model = sb3.PPO.load(model_path)
    elif rl_params["algo_name"] == "sac":
        model = sb3.SAC.load(model_path)
    elif rl_params["algo_name"] == "rad_ppo":
        model = RAD_PPO.load(model_path)
    elif rl_params["algo_name"] == "rad_sac":
        model = RAD_SAC.load(model_path)
    elif rl_params["algo_name"] == "MViTacRL_sac":
        model = MViTacRL_SAC.load(model_path)
    elif rl_params["algo_name"] == "MViTacRL_ppo":
        model = MViTacRL_PPO.load(model_path)
    else:
        sys.exit("Incorrect algorithm specified: {}.".format(rl_params["algo_name"]))

    # Seed the env
    if seed is not None:
        eval_env.reset()
        eval_env.seed(seed)

    # Evaluate and collect data
    episode_rewards, episode_lengths = collect_data_and_eval(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
    )

    print(np.mean(episode_rewards), np.std(episode_rewards), np.mean(episode_lengths), np.std(episode_lengths))

    eval_env.close()

    return np.mean(episode_rewards), np.std(episode_rewards), np.mean(episode_lengths), np.std(episode_lengths)


if __name__ == "__main__":
    # Data collection params
    n_eval_episodes = 1000
    deterministic = True
    show_gui = False
    show_tactile = False

    # Algorithm and environment settings
    algo_name = 'MViTacRL_sac'
    env_name = 'object_push-v0'
    obs_type = 'visuotactile_and_feature'

    # Model directory
    saved_model_dir = "saved_models/vtcon/20251103-111759"

    final_evaluation(
        saved_model_dir,
        n_eval_episodes,
        seed=seed,
        deterministic=deterministic,
        show_gui=show_gui,
        show_tactile=show_tactile,
    )