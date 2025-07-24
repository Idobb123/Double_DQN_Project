from agent import Agent
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DoubleAgent(Agent):
    """
    This class implements the optimization step done for a Double DQN agent.
    Calculates loss between current policy's Q values and target Q values.
    Uses the Double DQN algorithm to mitigate overestimation bias.

    The Double DQN algorithm uses the policy network to select actions and the target network to evaluate them.
    Qp(s, a) = r + γ * Qt(s', argmax(Qp(s', a'))
    where Qp is the policy network and Qt is the target network.
    """
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # Calculate target Q values (expected returns)
            best_actions = policy_dqn(new_states).argmax(dim=1)
            # Use the policy network to select actions and the target network to evaluate them
            target_q = rewards + (1 - terminations) * self.discount_factor * target_dqn(new_states).gather(dim=1, index=best_actions.unsqueeze(dim=1)).squeeze()

        # Calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters i.e. weights and biases