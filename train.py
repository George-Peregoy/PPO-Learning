import gymnasium
import config
from src.agent import Agent 
from src.utils import make_env
import numpy as np
import os
import json

def train(num_episodes, run_id, buffer_size, cfg):

    env_id = run_id.split("_")[0]
    env = make_env(env_id=env_id)

    root_dir = os.path.abspath(os.path.dirname(__file__))
    save_dir = os.path.join(root_dir, f'checkpoints/{run_id}')

    os.makedirs(os.path.join(root_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "metrics"), exist_ok=True)   

    agent = Agent(**cfg)

    state, info = env.reset()
    log_every = 20
    episode_rewards = []
    best_reward = 0
    episode_reward_log = [] # full history
    for i in range(num_episodes + 1):

        episode_reward = 0
        
        for _ in range(buffer_size):
            action, log_prob, value = agent.choose_action(state)

            next_state, reward, terminated, truncated, info = env.step(action.detach().numpy())
           
            done = 0
            if terminated == True or truncated == True:
                done = 1

            agent.buffer.store(state, action, log_prob, reward, value, done)

            episode_reward += reward

            if done:
                episode_rewards.append(episode_reward)
                episode_reward_log.append(episode_reward)
                episode_reward = 0
                state, info = env.reset()
            else:
                state = next_state

        agent.update()

        if episode_rewards:
            avg_reward = np.mean(episode_rewards)
            if avg_reward >= best_reward:
                agent.save(save_dir)
                best_reward = avg_reward

            if i % log_every == 0:
                print(f"Episode: {i} | avg reward: {avg_reward:.2f}")
                episode_rewards = []

    env.close()

    metrics = {
        "run_id" : run_id,
        **cfg,
        "rewards" : episode_reward_log
    }

    metric_dir = os.path.join(root_dir, "metrics", f"{run_id}.json")
    with open(metric_dir, 'w') as f:
        json.dump(metrics, f)

if __name__=="__main__":
    num_episodes = 2500
    env_id = config.ENV_ID

    config_list = [
        [config.BASELINE], 
        config.LR_SWEEP,
        config.LAYER_SWEEP,
        config.K_EPOCHS_SWEEP
    ]

    """for sweep in config_list:
        for setup in sweep:
            run_id = f'{env_id}_lr{setup["lr"]}_layers{setup["actor_layer_sizes"][1:-1]}_k{setup["k_epochs"]}'
            print(f'Starting run: {run_id}')
            train(num_episodes=num_episodes, run_id = run_id, buffer_size = config.BUFFER_SIZE, cfg=setup)
            print(f'Finished run: {run_id}\n')"""


    # candidates for final params run for 5000 to fully train
    best_candidates = [
        config.BEST_SWEEP
    ]

    """for sweep in best_candidates:
        for setup in sweep:
            run_id = f'{env_id}_lr{setup["lr"]}_layers{setup["actor_layer_sizes"][1:-1]}_k{setup["k_epochs"]}'
            print(f'Starting run: {run_id}')
            train(num_episodes=5000, run_id = run_id, buffer_size = config.BUFFER_SIZE, cfg=setup)
            print(f'Finished run: {run_id}\n')"""
    
    run_id = f'{config.ENV_ID}_lr{config.BEST_MODEL["lr"]}_layers{config.BEST_MODEL["actor_layer_sizes"][1:-1]}_k{config.BEST_MODEL["k_epochs"]}'
    print(f'Starting run: {run_id}')
    train(num_episodes=5000, run_id = run_id, buffer_size = config.BUFFER_SIZE, cfg=config.BEST_MODEL)
    print(f'Finished run: {run_id}\n')

