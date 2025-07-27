import torch as th
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.type_aliases import RolloutBufferSamples

class DoubleDQN(DQN):
    def _get_torch_target_q_values(self, replay_data: RolloutBufferSamples) -> th.Tensor:
        # Based on SB3 vanilla implementation
        with th.no_grad():
            # Q-values from target network
            next_q_values = self.q_net_target(replay_data.next_observations)
            if getattr(self, "double_dqn_enabled", True):
                # select argmax actions via online network
                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1)
                next_actions = next_actions.unsqueeze(-1)
                # evaluate selected actions via target network
                next_q = th.gather(next_q_values, dim=1, index=next_actions).squeeze(1)
            else:
                next_q, _ = next_q_values.max(dim=1)
        return next_q