import numpy as np
import torch
from torch.nn import ReLU, MSELoss
from torch.optim import Adam
from src.network import Network


class Agent:
    """
    PPO agent class.

    Attributes
    ----------
    device : torch.Device
        Device tensor operations are running on.
    state_dim : int
        Size of state vector.
    action_dim : int
        Size of action vector.
    lr : float
        Learning rate of actor and critic networks.
    actor_layer_sizes : list
        list of actor layer sizes. Example [state_dim, 256 64 action_dim].
    critic_layer_sizes : list
        List of critic layer sizes. Example [state_dim, 256, 64, 1].
    gamma : float
        Discount factor.
    epsilon : float
        Small radius used in clipping ratio of log probs.
    k_epochs : int
        Number of data collection steps before updating.
    gae_lambda : float
        Tuning param for advantages.

    Methods
    -------
    choose_action(state, deterministic)
        Chooses action given state, optionally deterministic.
    update()
        Updates actor and critic networks.
    """

    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 lr: float,
                 actor_layer_sizes: int,
                 critic_layer_sizes: int,
                 gamma: float,
                 epsilon: float,
                 k_epochs: int,
                 gae_lambda: float
                 ):
        
        # for tensors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # store params
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda

        # init buffer
        self.buffer = Buffer(self.device)

        # store log_std
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim)).to(self.device)

        # actor / critic networks
        self.actor_network = Network(
            layer_sizes = actor_layer_sizes, 
            lr = lr,
            parameters=self.log_std,
            activation_hidden = ReLU,
            optimizer = Adam
            )
        
        # loss is computed externally
        self.critic_network = Network(
            layer_sizes = critic_layer_sizes, 
            lr = lr,
            activation_hidden = ReLU,
            loss_fn = MSELoss,
            optimizer = Adam
            )
        
        # move to device
        self.actor_network = self.actor_network.to(self.device)
        self.critic_network = self.critic_network.to(self.device)

    def choose_action(self, state: np.ndarray, deterministic: bool=False):
        """
        Choose best action if training else sample from Normal distribution.

        Parameters
        ----------
        state : numpy.ndarray
            State vector from env step.
        deterministic : bool. Default is False.
            False if training else True.
        
        Returns
        -------
        action : torch.Tensor
            Action tensor.
        log_prob : torch.Tensor
            Log of probability of selecting that action.
        value : torch.Tensor
            Expected reward value.
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)

        with torch.no_grad():

            mean = self.actor_network(state) # chosen action
            value = self.critic_network(state) # expected value

            # if fully trained choose best
            if deterministic == True:
                return mean, None, value
            
            # if training
            dist = torch.distributions.Normal(mean, self.log_std.exp()) # convert log_std to std
        
            # sample from distribution
            action = dist.sample()

            log_prob = dist.log_prob(action).sum() # total log_prob of action
            return action, log_prob, value

    def update(self):
        
        # get stored data
        states, actions, log_probs, rewards, values, dones = self.buffer.get()

        # discount return
        R_t = 0
        returns = []
        for i in reversed(range(len(rewards))):
            R_t = rewards[i] + self.gamma * R_t * (1 - dones[i])
            returns.insert(0, R_t)

        # TD error
        values = torch.cat([values.squeeze(), torch.tensor([0.0])]).to(self.device) # add 0 for last value
        gae = 0
        advantages = []
        for i in reversed(range(len(rewards))):

            # find td error
            delta_t = rewards[i]+ self.gamma * values[i + 1] * (1-dones[i]) - values[i]

            # find advantages
            gae = delta_t + self.gamma * self.gae_lambda * (1-dones[i]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # update loop
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        for _ in range(self.k_epochs):
            
            # get new values
            mean = self.actor_network(states)
            dist = torch.distributions.Normal(mean, self.log_std.exp())
            new_log_probs = dist.log_prob(actions).sum(-1)
            current_values = self.critic_network(states).squeeze()

            # calc ratio for gae
            ratio = torch.exp(new_log_probs - log_probs)

            # clip ratio
            surr1 = ratio * advantages
            surr2 = torch.clip(ratio, 1-self.epsilon, 1 + self.epsilon) * advantages
            
            # get actor and critic loss
            actor_loss = -torch.min(surr1, surr2).mean() # negative to maximize 
            critic_loss = torch.nn.functional.mse_loss(current_values, returns)

            # update networks
            self.actor_network.update(loss=actor_loss)
            self.critic_network.update(loss=critic_loss)
        
        # clear buffer
        self.buffer.clear()

    def save(self, path: str):
        """
        Saves model params to path.

        Parameters
        ----------
        path : str
            Path params are being saved to.
        """
        torch.save(self.actor_network.state_dict(), f"{path}_actor.pt")
        torch.save(self.critic_network.state_dict(), f"{path}_critic.pt")

    def load(self, path: str):
        """
        Load params from path to agent.

        Parameters
        ----------
        path : str
            Path file name.
        """
        self.actor_network.load_state_dict(torch.load(f"{path}_actor.pt"))
        self.critic_network.load_state_dict(torch.load(f"{path}_critic.pt"))
    
class Buffer:
    """
    Generic buffer class.

    Attributes
    ----------
    states : list
        List of stored states.
    actions : list
        List of stored actions.
    log_probs : list
        List of stored log probs.
    rewards : list
        List of stored rewards.
    values : list
        List of stored values.
    dones : list
        List of stored dones.
    device : torch.Device
        Device torch operations are running on.

    Methods
    store(state, action, log_prob, reward, value, done)
        Stores data in respective list.
    clear()
        Resets stored lists.
    get()
        Returns lists as tuple of torch.Tensor32 in order states, actions, log_probs, rewards, values, dones.
    """

    def __init__(self, device):
        self.device = device
        
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store(self, 
              state: np.ndarray, 
              action: torch.Tensor, 
              log_prob: torch.Tensor, 
              reward: float, 
              value: torch.Tensor, 
              done: int):
        """
        Stores values for each step.

        state : np.ndarray
            State vector.
        action : torch.Tensor
            Chosen action.
        log_prob : torch.Tensor
            Log of probability of choosing that action.
        reward : float
            Reward for step.
        done : int
            1 if done else 0.
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32) # make sure torch.Tensor
        
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        
    def clear(self):
        """
        Resets buffer to initial state.
        """
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def get(self):
        """
        Returns
        -------
        states : torch.Tensor
            Stored states as tensor. 
        actions : torch.Tensor
            Stored actions as tensor.
        log_probs : torch.Tensor
            Stored log prob as tensor.
        rewards : torch.Tensor
            Stored rewards as tensor.
        values : torch.Tensor
            Stored values as tensor.
        dones: torch.Tensor
            Stored dones as tensor.
        """
        if self.device:
            return(
                torch.stack(self.states).to(self.device),
                torch.stack(self.actions).to(self.device),
                torch.stack(self.log_probs).to(self.device),
                torch.tensor(self.rewards, dtype=torch.float32).to(self.device),
                torch.stack(self.values).to(self.device),
                torch.tensor(self.dones, dtype=torch.float32).to(self.device)
            )
        else:
            return (
                torch.stack(self.states),
                torch.stack(self.actions),
                torch.stack(self.log_probs),
                torch.tensor(self.rewards, dtype=torch.float32),
                torch.stack(self.values),
                torch.tensor(self.dones, dtype=torch.float32)
            )