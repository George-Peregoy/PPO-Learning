from src.utils import make_env, get_env_dims

ENV_ID = "HalfCheetah-v5"
env = make_env(env_id=f"{ENV_ID}")
STATE_DIM, ACTION_DIM = get_env_dims(env)
STATE_DIM, ACTION_DIM = int(STATE_DIM), int(ACTION_DIM)
env.close()

BUFFER_SIZE = 4096

BASELINE = {
    "state_dim" : STATE_DIM,
    "action_dim": ACTION_DIM,
    "lr": 3e-4,
    "actor_layer_sizes": [STATE_DIM, 256, 256, ACTION_DIM],
    "critic_layer_sizes": [STATE_DIM, 256, 256, 1],
    "gamma": 0.99,
    "epsilon": 0.2,
    "k_epochs": 10,
    "gae_lambda": 0.95,
}

LR_SWEEP = [
    {**BASELINE, "lr" : 1e-4},
    {**BASELINE, "lr" : 1e-3},
]

LAYER_SWEEP = [
    {**BASELINE, "actor_layer_sizes": [STATE_DIM, 64, 64, ACTION_DIM], "critic_layer_sizes": [STATE_DIM, 64, 64, 1]},
    {**BASELINE, "actor_layer_sizes": [STATE_DIM, 128, 128, ACTION_DIM], "critic_layer_sizes": [STATE_DIM, 128, 128, 1]},
]

K_EPOCHS_SWEEP = [
    {**BASELINE, "k_epochs" : 15},
    {**BASELINE, "k_epochs" : 20}
]

# Found params visually after comparing sweeps
BEST_SWEEP = [
    {**BASELINE, "lr" : 1e-4, "k_epochs" : 20, "actor_layer_sizes": [STATE_DIM, 64, 64, ACTION_DIM], "critic_layer_sizes": [STATE_DIM, 64, 64, 1]},
    {**BASELINE, "lr" : 1e-4, "k_epochs" : 20, "actor_layer_sizes": [STATE_DIM, 256, 256, ACTION_DIM], "critic_layer_sizes": [STATE_DIM, 256, 256, 1]}
]

BEST_MODEL = {**BASELINE, "lr" : 1e-4, "k_epochs" : 20, "actor_layer_sizes": [STATE_DIM, 256, 256, ACTION_DIM], "critic_layer_sizes": [STATE_DIM, 256, 256, 1]}