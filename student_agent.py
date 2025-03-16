# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from obs_to_state import obs_to_state

def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

def get_action(obs):

    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys.
    #       Otherwise, even if your agent performs well in training, it may fail during testing.

    state = obs_to_state(obs)

    with open('q_table.pkl', 'rb') as file:
        q_table = pickle.load(file)

    while state not in q_table:
        target_dir, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = state
        target_dir = (sign(target_dir[0]) * (abs(target_dir[0]) - 1),
                      sign(target_dir[1]) * (abs(target_dir[1]) - 1))
        state = (target_dir, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
        if target_dir[0] == 0 and target_dir[1] == 0:
            break

    if state in q_table:
        return max(q_table[state], key=q_table[state].get)

    return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.
