from torchrl.modules import Actor, MLP, ValueOperator
from torchrl.objectives import DDPGLoss
from torchrl.envs.transforms import ActionMask, TransformedEnv


from env import LastSurvivors

base_env = LastSurvivors()
env = TransformedEnv(base_env, ActionMask())

n_obs = 4
n_act = 1
actor = Actor(MLP(in_features=n_obs, out_features=n_act, num_cells=[32, 32]))
value_net = ValueOperator(
    MLP(in_features=n_obs + n_act, out_features=1, num_cells=[32, 32]),
    in_keys=["choices", "action"],
)

ddpg_loss = DDPGLoss(actor_network=actor, value_network=value_net)

rollout = env.rollout(max_steps=100, policy=actor)
loss_vals = ddpg_loss(rollout)