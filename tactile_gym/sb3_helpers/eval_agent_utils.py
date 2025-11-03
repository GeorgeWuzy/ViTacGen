import os
import sys
import numpy as np
import cv2

import stable_baselines3 as sb3
import tactile_gym.rl_envs # register env
from sb3_contrib import RAD_PPO, RAD_SAC
from tactile_gym.sb3_helpers.custom_policies import MViTacRL_SAC, MViTacRL_PPO
from tactile_gym.sb3_helpers.rl_utils import make_eval_env
from tactile_gym.utils.general_utils import load_json_obj


def eval_and_save_vid(  
    model, env, saved_model_dir, n_eval_episodes=10, deterministic=True, render=False, save_vid=False, take_snapshot=False
):  
    if save_vid:  
        record_every_n_frames = 1  
        render_img = env.render(mode="rgb_array")  
        render_img_size = (render_img.shape[1], render_img.shape[0])  
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
        out = cv2.VideoWriter(  
            os.path.join(saved_model_dir, "evaluated_policy.mp4"),  
            fourcc,  
            24.0,  
            render_img_size,  
        )  

    if take_snapshot:  
        render_img = env.render(mode="rgb_array")  
        render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)  
        cv2.imwrite(os.path.join(saved_model_dir, "env_snapshot.png"), render_img)  

    episode_rewards, episode_lengths = [], []  
    success_count = 0 
    distance_errors = []

    for idx in range(n_eval_episodes):  
        print(idx)  
        obs = env.reset()  
        done, state = False, None  
        episode_reward = 0.0  
        episode_length = 0  
        final_goal_xy = None  
        final_object_xy = None  

        while not done:  
            action, state = model.predict(obs, state=state, deterministic=deterministic)  
            obs, reward, done, info = env.step(action)  

            episode_reward += reward  
            episode_length += 1  

            # render visual + tactile observation  
            if render:  
                render_img = env.render(mode="rgb_array")  
            else:  
                render_img = None  

            # write rendered image to mp4  
            # use record_every_n_frames to reduce size sometimes  
            if save_vid and episode_length % record_every_n_frames == 0:  
                # warning to enable rendering  
                if render_img is None:  
                    sys.exit("Must be rendering to save video")  

                render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)  
                out.write(render_img)  

            if take_snapshot:  
                render_img = env.render(mode="rgb_array")  
                render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)  
                
                left_img = render_img[:, :512]
                
                cv2.imwrite(os.path.join(saved_model_dir, "env_snapshot.png"), left_img)  
                quit()

        final_goal_xy = info[0].get("goal_xy", None)  
        final_object_xy = info[0].get("object_xy", None)  

        if final_goal_xy is not None and final_object_xy is not None:  
            distance_error = np.sqrt((final_goal_xy[0] - final_object_xy[0])**2 + (final_goal_xy[1] - final_object_xy[1])**2)  
            if distance_error < 100:
                distance_errors.append(distance_error)
        print(distance_error)
        if distance_error < 40.0:  # Adjust success threshold here
            success_count += 1  

        episode_rewards.append(episode_reward)  
        episode_lengths.append(episode_length)  
        
    if save_vid:  
        out.release()  

    accuracy = success_count / n_eval_episodes * 100  

    mean_distance_error = np.mean(distance_errors) if distance_errors else 0  

    return episode_rewards, episode_lengths, mean_distance_error, accuracy

def final_evaluation(  
    saved_model_dir,  
    n_eval_episodes,  
    seed=None,  
    deterministic=True,  
    show_gui=True,  
    show_tactile=True,  
    render=False,  
    save_vid=False,  
    take_snapshot=False
):  

    rl_params = load_json_obj(os.path.join(saved_model_dir, "rl_params"))  
    algo_params = load_json_obj(os.path.join(saved_model_dir, "algo_params"))  
    rl_params["image_size"] = [512, 512]  
    # create the evaluation env  
    eval_env = make_eval_env(  
        rl_params["env_name"],  
        rl_params,  
        show_gui=show_gui,  
        show_tactile=show_tactile,  
    )  

    # load the trained model  
    model_path = os.path.join(saved_model_dir, "trained_models", "best_model.zip")  

    # create the model with hyper params  
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
        sys.exit("Incorrect algorithm specified: {}.".format(algo_name))  

    # seed the env  
    if seed is not None:  
        eval_env.reset()  
        eval_env.seed(seed)  

    # evaluate the trained agent  
    episode_rewards, episode_lengths, mean_distance_error, accuracy = eval_and_save_vid(  
        model,  
        eval_env,  
        saved_model_dir,  
        n_eval_episodes=n_eval_episodes,  
        deterministic=deterministic,  
        save_vid=save_vid,  
        render=render,  
        take_snapshot=take_snapshot
    )  

    print(saved_model_dir)
    print("Mean Reward:", np.mean(episode_rewards))  
    print("Reward Std:", np.std(episode_rewards))  
    print("Mean Episode Length:", np.mean(episode_lengths))  
    print("Episode Length Std:", np.std(episode_lengths))  
    print("Mean Distance Error:", mean_distance_error)  
    print("Accuracy:", accuracy)  

    eval_env.close()  

    return

if __name__ == "__main__":  

    # Adjust eval parameters here
    n_eval_episodes = 100
    seed = int(0)  
    deterministic = True
    show_gui = False
    show_tactile = False
    render = False
    save_vid = False
    take_snapshot = False

    # algo_name = 'sac'  
    algo_name = 'MViTacRL_sac'  
    env_name = 'object_push-v0'  
    obs_type = 'visuotactile_and_feature'  
    saved_model_dir = "saved_models/vtcon/20251103-111759"
    # saved_model_dir = "saved_models/vitacgen"

    final_evaluation(  
        saved_model_dir,  
        n_eval_episodes,  
        seed=seed,  
        deterministic=deterministic,  
        show_gui=show_gui,  
        show_tactile=show_tactile,  
        render=render,  
        save_vid=save_vid,  
        take_snapshot=take_snapshot
    )