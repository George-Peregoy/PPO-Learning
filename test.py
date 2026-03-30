import gymnasium as gym
import config
from src.agent import Agent 
import os
import numpy as np

def test(num_episodes, run_id):
    
    root_dir = os.path.abspath(os.path.dirname(__file__))
    load_dir = os.path.join(root_dir, f'checkpoints/{run_id}')
    video_dir = os.path.join(root_dir, f"videos/{run_id}")

    env = gym.wrappers.RecordVideo(
    gym.make(config.ENV_ID, render_mode="rgb_array"),
    video_folder=video_dir,
    episode_trigger=lambda ep: ep == 0 
    )

    cfg = config.BEST_SWEEP[1]
    agent = Agent(**cfg)

    agent.load(load_dir)

    avg_reward = []

    for _ in range(num_episodes):

        state, info = env.reset()
        episode_reward = 0
        done = False

        while not done:

            action, _, _ = agent.choose_action(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action.detach().numpy())
            episode_reward += reward
            done = terminated or truncated

        avg_reward.append(episode_reward)
        print(f"Episode return: {episode_reward:.2f}")

    print(f"avg reward: {np.mean(avg_reward)}")
    env.close()

if __name__=="__main__":
    num_episodes = 30
    run_id = f'{config.ENV_ID}_lr{config.BEST_MODEL["lr"]}_layers{config.BEST_MODEL["actor_layer_sizes"][1:-1]}_k{config.BEST_MODEL["k_epochs"]}'
    test(num_episodes=num_episodes, run_id=run_id)