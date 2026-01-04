import numpy as np

class SpinQubitSensorEnv:
    def __init__(self, B_values, T2=1.0, max_steps=3):
        self.B_values = np.array(B_values, dtype=float)  # discrete possible B values
        self.N = len(B_values)         # number of discrete states for B
        self.T2 = T2                   # dephasing time constant
        self.max_steps = max_steps     # measurement count per episode
        self.times = None              # list of measurement times (actions)
        self.reset()                   # initialize environment state

    def set_times(self, times):
        """Define the discrete measurement times corresponding to each action index."""
        self.times = np.array(times, dtype=float)
    
    def reset(self):
        """Start a new episode: sample a true B, reset belief to uniform."""
        self.true_idx = np.random.randint(0, self.N)            # index of true B in the list
        self.true_B = self.B_values[self.true_idx]              # actual magnetic field for this episode
        self.belief = np.ones(self.N) / self.N                  # uniform prior
        self.step_count = 0
        # State includes current belief distribution and remaining steps (optional)
        return self._get_state()
    
    def _get_state(self):
        """Construct the state vector (belief + remaining steps)."""
        remaining_steps = self.max_steps - self.step_count
        # We append the remaining step count (normalized, e.g. /max_steps if needed) to the belief vector
        return np.concatenate([self.belief, [remaining_steps]])
    
    def step(self, action):
        """Simulate one measurement step with the chosen action (time index)."""
        assert self.times is not None, "Measurement times not set. Call set_times(...) first."
        tau = float(self.times[action])               # selected measurement duration
        # Quantum sensor evolution: compute outcome probabilities
        # P(+X outcome) = 0.5 * (1 + e^{-tau/T2} * cos(B * tau))
        exp_decay = np.exp(-tau / self.T2)
        cos_phase = np.cos(self.true_B * tau)
        p_plus = 0.5 * (1 + exp_decay * cos_phase)    # probability of getting outcome +1 (X-basis)
        # Sample a measurement outcome according to this probability
        outcome = 1 if np.random.rand() < p_plus else 0  # 1 for +X outcome, 0 for -X outcome
        
        # Bayesian belief update:
        # Calculate likelihoods P(outcome | B_i) for each candidate B_i
        likelihoods = 0.5 * (1 + np.exp(-tau/self.T2) * np.cos(self.B_values * tau))
        if outcome == 0:
            likelihoods = 1 - likelihoods  # if we got the "-X" outcome, use 1 - p_plus probabilities
        # Update posterior belief via elementwise multiplication and normalization
        prior = self.belief
        unnorm_post = prior * likelihoods
        if unnorm_post.sum() == 0:  
            # Numerical safety: if all probabilities zero (unlikely), keep prior
            post = prior
        else:
            post = unnorm_post / unnorm_post.sum()
        self.belief = post
        self.step_count += 1
        
        # Calculate reward: information gain (reduction in entropy)
        def entropy(p_dist):
            mask = p_dist > 0
            return -np.sum(p_dist[mask] * np.log2(p_dist[mask]))
        prev_entropy = entropy(prior)
        new_entropy = entropy(post)
        reward = prev_entropy - new_entropy  # positive if uncertainty decreased
        
        # If this was the last measurement, provide final accuracy reward
        done = (self.step_count >= self.max_steps)
        if done:
            # Determine final estimate (most likely B)
            est_idx = int(np.argmax(self.belief))
            if est_idx == self.true_idx:
                reward += 1.0   # bonus for correct identification
            else:
                reward += 0.0   # (or a small negative penalty for wrong, e.g. reward -= 0.5)
        
        # Construct next state (or None if done)
        next_state = self._get_state() if not done else None
        return next_state, reward, done, {"outcome": outcome, "true_B": self.true_B}

import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        logits = self.fc2(x)    # raw scores for each action
        return logits           # (we'll apply softmax via PyTorch distributions when sampling)



import torch.optim as optim

# Initialize environment and policy
B_values = [0.5, 1.0, 1.5, 2.0, 2.5]           # possible magnetic field values (discrete)
env = SpinQubitSensorEnv(B_values, T2=1.0, max_steps=3)
env.set_times([0.1, 0.5, 1.0, 2.0, 3.0])       # define 5 possible measurement times (seconds, for example)
state_dim = env.N + 1                          # belief length + 1
action_dim = len(env.times)                    # number of discrete actions
policy = PolicyNet(state_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# Training parameters
num_episodes = 5000
gamma = 1.0  # discount factor (can be 1 for episodic tasks focusing on final outcome)

for episode in range(num_episodes):
    state = env.reset()  
    state = torch.tensor(state, dtype=torch.float32)
    log_probs = []
    rewards = []
    # Generate an episode
    done = False
    while not done:
        # Get action probabilities from policy
        logits = policy(state)  
        dist = torch.distributions.Categorical(logits=logits)  # categorical distribution over actions
        action = dist.sample()                                 # sample an action index
        log_prob = dist.log_prob(action)                       # log π(a|s)
        next_state, reward, done, info = env.step(int(action.item()))
        
        # Record the log-prob and reward
        log_probs.append(log_prob)
        rewards.append(reward)
        
        # Move to next state
        if next_state is not None:
            state = torch.tensor(next_state, dtype=torch.float32)
    # Episode ended. Compute returns and update policy.
    # Calculate discounted returns for each step (here gamma=1, so it's just cumulative future reward from that step)
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    # Optionally normalize returns for stability
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # Policy gradient: maximize E[return * log_prob] -> minimize -(return * log_prob)
    loss = 0.0
    for log_prob, R in zip(log_probs, returns):
        loss += -log_prob * R
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # (Optional) logging
    if episode % 500 == 0:
        total_reward = sum(rewards)
        print(f"Episode {episode}: total reward = {total_reward:.3f}")

# After training, evaluate the policy
test_episodes = 1000
correct_count = 0
for _ in range(test_episodes):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    while not done:
        logits = policy(state)
        action = torch.argmax(logits).item()  # choose the action with highest probability (greedy)
        next_state, reward, done, info = env.step(action)
        if next_state is not None:
            state = torch.tensor(next_state, dtype=torch.float32)
    # After episode, check if final estimate was correct
    est_idx = int(np.argmax(env.belief))
    if est_idx == env.true_idx:
        correct_count += 1
accuracy = correct_count / test_episodes
print(f"Policy accuracy over {test_episodes} test episodes: {accuracy*100:.1f}%")

