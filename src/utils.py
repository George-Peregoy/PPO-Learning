import gymnasium as gym
import numpy as np

def make_env(env_id: str = 'HalfCheetah-v5', render: bool = False):
    """
    Makes gymnasium environment.

    Parameters
    ----------
    env_id : str 
        Name of gym env.
    render : bool. Default is False.
        True if human render else False.
    
    Returns
    -------
    env : gymnasium environment
    """
    env = gym.make(env_id, render_mode='human' if render else None)
    return env

def get_env_dims(env):
    """
    Gets action and state dim from environment.

    Parameters
    ----------
    env : gymnasium environment
    
    Returns
    -------
    state_dim : int
        Size of state vector.
    action_dim : int
        Size of action vector
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    return state_dim, action_dim

def smooth(rewards, window=50):
    """
    Smooths reward signal using convolution, helps remove noise.

    Parameters
    ----------
    rewards : list
        History of rewards.
    window : int
        Range of convolution.

    Returns
    -------
    denoised_rewards : numpy.ndarray
        Smoothed rewards.
    """
    return np.convolve(rewards, np.ones(window)/window, mode='valid')