import torch
from torchrl.data.tensor_specs import DiscreteTensorSpec, BinaryDiscreteTensorSpec, UnboundedContinuousTensorSpec, CompositeSpec, UnboundedDiscreteTensorSpec
from torchrl.envs.transforms import ActionMask, TransformedEnv
from torchrl.envs.common import EnvBase

from utils import screenshot, get_choices

class MaskedEnv(EnvBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_spec = DiscreteTensorSpec(4)
        self.state_spec = CompositeSpec( # Encode Level Information Here
            action_mask=BinaryDiscreteTensorSpec(4, dtype=torch.bool))
        self.observation_spec = CompositeSpec(choices=UnboundedContinuousTensorSpec(4)) # choices, health info?, level
        self.reward_spec = UnboundedContinuousTensorSpec(1)

    def _reset(self, tensordict=None):
        td = self.observation_spec.rand()
        td.update(torch.ones_like(self.state_spec.rand()))
        return td # return the state

    def _step(self, data):
        # sc = screenshot(save=True)
        choices = get_choices()
        td = self.observation_spec.rand()
        mask = data.get("action_mask")
        action = data.get("action")
        mask = mask.scatter(-1, action.unsqueeze(-1), 0)

        td.set("action_mask", mask)
        td.set("reward", self.reward_spec.rand())
        td.set("done", ~mask.any().view(1))
        return td

    def _set_seed(self, seed):
        return seed

torch.manual_seed(0)
base_env = MaskedEnv()
env = TransformedEnv(base_env, ActionMask())
r = env.rollout(10)
env = TransformedEnv(base_env, ActionMask())
r = env.rollout(10)
r["action_mask"]

