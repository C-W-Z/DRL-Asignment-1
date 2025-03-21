# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

def get_vector(from_pos, to_pos):
    # if from_pos[0] < to_pos[0]: return 0  # Down
    # if from_pos[0] > to_pos[0]: return 1  # Up
    # if from_pos[1] < to_pos[1]: return 2  # Right
    # if from_pos[1] > to_pos[1]: return 3  # Left
    # return random.choice([0, 1, 2, 3])  # 亂走
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

class RuleAgent:
    def __init__(self):
        self.memory = {
            "passenger_pos": None,
            "destination_pos": None,
            "has_passenger": False,
            "visited_corners": set()
        }

    def update_memory(self, obs, action):
        taxi_x, taxi_y = obs[:2]  # 假設 obs[0] = x, obs[1] = y

        # 如果進入角落，記錄它
        if (taxi_x, taxi_y) in [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]:
            self.memory["visited_corners"].add((taxi_x, taxi_y))

        # 如果 pickup 成功，記錄乘客位置
        if action == 4 and obs[14] and (taxi_x, taxi_y) in [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]:
            self.memory["passenger_pos"] = (taxi_x, taxi_y)
            self.memory["has_passenger"] = True

        # 如果 dropoff 成功，記錄目的地位置
        if obs[15] and (taxi_x, taxi_y) in [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]:
            self.memory["destination_pos"] = (taxi_x, taxi_y)
            self.memory["has_passenger"] = False

    def move_towards(self, x, y, target_x, target_y):
        if x < target_x: return 0  # Down
        if x > target_x: return 1  # Up
        if y < target_y: return 2  # Right
        if y > target_y: return 3  # Left
        return random.choice([0, 1, 2, 3])  # 亂走

    def choose_action(self, obs):
        taxi_x, taxi_y = obs[:2]

        close = taxi_close_to_station(obs)
        if close is not None:
            if obs[14]:
                self.memory["passenger_pos"] = close
                # print("passenger_pos", close)
            if obs[15]:
                self.memory["destination_pos"] = close
                # print("destination_pos", close)
            self.memory["visited_corners"].add(close)

        # 如果還沒找到乘客和目的地，優先探索四個角落
        if self.memory["passenger_pos"] is None or self.memory["destination_pos"] is None:
            corners = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]
            unexplored = [c for c in corners if c not in self.memory["visited_corners"]]
            if unexplored:
                target_x, target_y = unexplored[0]
                return self.move_towards(taxi_x, taxi_y, target_x, target_y)

        # 接乘客
        if obs[14] and (taxi_x, taxi_y) in [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])] and not self.memory["has_passenger"]:
            self.memory["has_passenger"] = True
            # print("pickup")
            return 4

        # 前往乘客所在地
        if not self.memory["has_passenger"]:
            target_x, target_y = self.memory["passenger_pos"]
            if (taxi_x, taxi_y) == (target_x, target_y):
                return 4
            # print("target passenger:", target_x, target_y, taxi_x, taxi_y)
            return self.move_towards(taxi_x, taxi_y, target_x, target_y)

        # 放下乘客
        if obs[15] and (taxi_x, taxi_y) in [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]:
            return 5

        # 接到乘客後，前往目的地
        if self.memory["has_passenger"]:
            target_x, target_y = self.memory["destination_pos"]
            if (taxi_x, taxi_y) == (target_x, target_y):
                return 5
            # print("target destination:", target_x, target_y, taxi_x, taxi_y)
            return self.move_towards(taxi_x, taxi_y, target_x, target_y)

        return random.choice([0, 1, 2, 3])

class QAgent:
    def __init__(self):
        self.reset()
        self.q_table = {}

    def reset(self):
        self.passenger_pos = None
        self.destination_pos = None
        self.has_passenger = False
        self.has_first_picked_up = False
        self.visited_corners = set()

    def load_q_table(self, path='q_table.pkl'):
        with open(path, 'rb') as file:
            self.q_table = pickle.load(file)

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

        # 如果還沒找到乘客和目的地，優先探索四個角落
        if self.passenger_pos is None:
            unexplored = [c for c in stations if c not in self.visited_corners]
            if unexplored:
                target_dir = get_vector(taxi_pos, unexplored[0])
                return (
                    target_dir,
                    obstacle_north, obstacle_south, obstacle_east, obstacle_west,
                    at_passenger, at_destination,
                    self.has_passenger
                )

        # 前往乘客所在地
        if not self.has_passenger and self.passenger_pos:
            # print("target passenger:", target_x, target_y, taxi_x, taxi_y)
            target_dir = get_vector(taxi_pos, self.passenger_pos)
            return (
                target_dir,
                obstacle_north, obstacle_south, obstacle_east, obstacle_west,
                at_passenger, at_destination,
                self.has_passenger
            )

        if self.destination_pos is None:
            unexplored = [c for c in stations if c not in self.visited_corners]
            if unexplored:
                target_dir = get_vector(taxi_pos, unexplored[0])
                return (
                    target_dir,
                    obstacle_north, obstacle_south, obstacle_east, obstacle_west,
                    at_passenger, at_destination,
                    self.has_passenger
                )

        if self.has_passenger and self.destination_pos:
            # print("target destination:", target_x, target_y, taxi_x, taxi_y)
            target_dir = get_vector(taxi_pos, self.destination_pos)
            return (
                target_dir,
                obstacle_north, obstacle_south, obstacle_east, obstacle_west,
                at_passenger, at_destination,
                self.has_passenger
            )

        target_dir = (0, 0)
        return (
            target_dir,
            obstacle_north, obstacle_south, obstacle_east, obstacle_west,
            at_passenger, at_destination,
            self.has_passenger
        )

    def get_action(self, state):
        if state in self.q_table:
            return max(self.q_table[state], key=self.q_table[state].get)
        return random.choice([0, 1, 2, 3])

    def update_has_passenger(self, obs, state, action):
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

    def update_q_table(self, state, next_state, action, alpha, reward, gamma):
        self.q_table[state][action] += alpha * (reward + gamma * max(self.q_table[next_state].values()) - self.q_table[state][action])

    def init_state_in_q_table(self, state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in range(6)} # 6 actions

    def save_q_table(self):
        with open('q_table.pkl', 'wb') as file:
            pickle.dump(self.q_table, file, protocol=pickle.HIGHEST_PROTOCOL)

# agent = RuleAgent()
agent = QAgent()
agent.load_q_table('q_table.pkl')

def get_action(obs):

    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys.
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    state = agent.obs_to_state(obs)
    action = agent.get_action(state)
    agent.update_has_passenger(obs, state, action)
    return action

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
