# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

eps = np.finfo(np.float32).eps.item()

def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

def get_vector(from_pos, to_pos):
    return (sign(to_pos[0] - from_pos[0]), sign(to_pos[1] - from_pos[1]))

def taxi_close_to_station(obs):
    r, c = obs[:2]
    s = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]
    for sr, sc in s:
        if r == sr and c == sc:
            return (sr, sc)
        elif r == sr and (c == sc - 1 or c == sc + 1):
            return (sr, sc)
        elif (r == sr - 1 or r == sr + 1) and c == sc:
            return (sr, sc)
    return None

class PolicyAgent(nn.Module):
    def __init__(self):
        super(PolicyAgent, self).__init__()
        self.init_policy()
        self.reset()

    def init_policy(self):
        self.state_size = (3, 3, 2, 2, 2, 2, 2, 2, 2)
        self.action_size = 6

        self.affine1 = nn.Linear(9, 32)
        self.dropout = nn.Dropout(p=0.05)
        self.affine2 = nn.Linear(32, self.action_size)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.saved_log_probs = []
        self.rewards = []

    def reset(self):
        self.passenger_pos = None
        self.destination_pos = None
        self.has_passenger = False
        self.has_first_picked_up = False
        self.visited_corners = set()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

    def get_action(self, state):
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        probs = self(state_tensor)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def update(self, gamma):
        R = 0
        policy_loss = []
        returns = deque()
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.appendleft(R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

    def clean_update(self):
        del self.rewards[:]
        del self.saved_log_probs[:]

    def obs_to_state(self, obs):
        stations = [[0,0] for _ in range(4)]
        taxi_row, taxi_col, stations[0][0], stations[0][1], stations[1][0], stations[1][1], stations[2][0], stations[2][1], stations[3][0], stations[3][1], obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
        stations = [(r, c) for [r, c] in stations]
        taxi_pos = (taxi_row, taxi_col)

        if self.has_passenger:
            self.passenger_pos = taxi_pos

        close = taxi_close_to_station(obs)
        if close is not None:
            self.visited_corners.add(close)
            if passenger_look and not self.has_passenger and not self.has_first_picked_up:
                self.passenger_pos = close
            if destination_look:
                self.destination_pos = close

        at_passenger = (self.passenger_pos is not None) and (self.passenger_pos == taxi_pos)
        at_destination = (self.destination_pos is not None) and (self.destination_pos == taxi_pos)

        if self.passenger_pos is None:
            unexplored = [c for c in stations if c not in self.visited_corners]
            if unexplored:
                target_dir = get_vector(taxi_pos, unexplored[0])
                return (
                    target_dir[0] + 1, target_dir[1] + 1,
                    int(obstacle_north), int(obstacle_south), int(obstacle_east), int(obstacle_west),
                    int(at_passenger), int(at_destination), int(self.has_passenger)
                )

        if not self.has_passenger and self.passenger_pos:
            target_dir = get_vector(taxi_pos, self.passenger_pos)
            return (
                target_dir[0] + 1, target_dir[1] + 1,
                int(obstacle_north), int(obstacle_south), int(obstacle_east), int(obstacle_west),
                int(at_passenger), int(at_destination), int(self.has_passenger)
            )

        if self.destination_pos is None:
            unexplored = [c for c in stations if c not in self.visited_corners]
            if unexplored:
                target_dir = get_vector(taxi_pos, unexplored[0])
                return (
                    target_dir[0] + 1, target_dir[1] + 1,
                    int(obstacle_north), int(obstacle_south), int(obstacle_east), int(obstacle_west),
                    int(at_passenger), int(at_destination), int(self.has_passenger)
                )

        if self.has_passenger and self.destination_pos:
            # print("target destination:", target_x, target_y, taxi_x, taxi_y)
            target_dir = get_vector(taxi_pos, self.destination_pos)
            return (
                target_dir[0] + 1, target_dir[1] + 1,
                int(obstacle_north), int(obstacle_south), int(obstacle_east), int(obstacle_west),
                int(at_passenger), int(at_destination), int(self.has_passenger)
            )

        target_dir = (0, 0)
        return (
            target_dir[0] + 1, target_dir[1] + 1,
            int(obstacle_north), int(obstacle_south), int(obstacle_east), int(obstacle_west),
            int(at_passenger), int(at_destination), int(self.has_passenger)
        )

    def update_after_get_action(self, obs, state, action):
        taxi_row, taxi_col, _, _, _, _, _, _, _, _, _, _, _, _, passenger_look, _ = obs
        taxi_pos = (taxi_row, taxi_col)

        if passenger_look and not self.has_passenger and action == 4 and taxi_pos == self.passenger_pos:
            self.has_passenger = True
            self.has_first_picked_up = True
            # print("pickup")
        elif self.has_passenger and action == 5:
            self.has_passenger = False
            # print("dropoff")
            if state[2]:
                self.reset()

def save_checkpoint(model, filename='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, filename='checkpoint.pth'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

agent = PolicyAgent()
load_checkpoint(agent)
agent.eval()

def get_action(obs):

    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys.
    #       Otherwise, even if your agent performs well in training, it may fail during testing.

    state = agent.obs_to_state(obs)
    action = agent.get_action(state)
    agent.update_after_get_action(obs, state, action)
    return action
