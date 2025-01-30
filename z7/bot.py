"""
==========================================
Deep Q-Network Agent for NameThisGame (Atari)
Creator:
Michał Cichowski s20695
==========================================
To run program install:
pip install gymnasium[atari] torch numpy ale-py
==========================================
Usage:
python bot.py
==========================================
"""

import os
import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Hyperparameters for the DQN agent
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 32
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10

# Deep Q-Network model
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Automatically determine the size of the flattened layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_out_size = self.conv(dummy_input).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x.view(x.size(0), -1))
        return x

# DQN agent
class DQNAgent:
    def __init__(self, state_shape, n_actions):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_shape, n_actions).to(self.device)
        self.target_model = DQN(state_shape, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.n_actions = n_actions

    def select_action(self, state):
        # Choose an action based on epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        # Train the model using a batch of experiences
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # Update the target network with the main network's weights
        self.target_model.load_state_dict(self.model.state_dict())

# Load the game ROM and initialize environment
def load_namethisgame():
    rom_path = os.path.abspath("z7/Name This Game.bin")  # Full path to the ROM
    roms_folder = os.path.expanduser("~/.ale/roms")  # ALE ROMs folder

    # Check if the ROM exists in the ALE ROMs directory
    if not os.path.exists(os.path.join(roms_folder, "Name This Game.bin")):
        print(f"Copying ROM to {roms_folder}...")
        os.makedirs(roms_folder, exist_ok=True)
        os.system(f'copy "{rom_path}" "{roms_folder}"' if os.name == "nt" else f'cp "{rom_path}" "{roms_folder}"')

    # Create Gymnasium environment for the game
    env = gym.make("ALE/NameThisGame-v5", render_mode="human")
    return env

# Main training loop
def train_agent():
    env = load_namethisgame()
    state_shape = (3, 210, 160)
    n_actions = env.action_space.n
    agent = DQNAgent(state_shape, n_actions)

    num_episodes = 500
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.transpose(state, (2, 0, 1)) / 255.0
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.transpose(next_state, (2, 0, 1)) / 255.0

            agent.store_transition(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward

        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

if __name__ == "__main__":
    train_agent()
