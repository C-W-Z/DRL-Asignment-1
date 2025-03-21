import gym
import numpy as np
import importlib.util
import time
from IPython.display import clear_output
import random
from obs_to_state import obs_to_state
import pickle
# This environment allows you to verify whether your program runs correctly during testing,
# as it follows the same observation format from `env.reset()` and `env.step()`.
# However, keep in mind that this is just a simplified environment.
# The full specifications for the real testing environment can be found in the provided spec.
#
# You are free to modify this file to better match the real environment and train your own agent.
# Good luck!


class SimpleTaxiEnv():
    def __init__(self, grid_size=5, fuel_limit=50):
        """
        Custom Taxi environment supporting different grid sizes.
        """
        self.grid_size = grid_size
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.passenger_picked_up = False

        self.stations = [(0, 0), (0, self.grid_size - 1), (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)]
        self.passenger_loc = None

        self.obstacles = set()  # No obstacles in simple version
        self.destination = None

    def reset(self):
        """Reset the environment, ensuring Taxi, passenger, and destination are not overlapping obstacles"""
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False


        available_positions = [
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
            if (x, y) not in self.stations and (x, y) not in self.obstacles
        ]

        self.taxi_pos = random.choice(available_positions)

        self.passenger_loc = random.choice([pos for pos in self.stations])


        possible_destinations = [s for s in self.stations if s != self.passenger_loc]
        self.destination = random.choice(possible_destinations)

        return self.get_state(), {}

    def step(self, action):
        """Perform an action and update the environment state."""
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0
        if action == 0 :  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1


        if action in [0, 1, 2, 3]:  # Only movement actions should be checked
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -=5
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
        else:
            if action == 4:  # PICKUP
                if self.taxi_pos == self.passenger_loc:
                    self.passenger_picked_up = True
                    self.passenger_loc = self.taxi_pos
                else:
                    reward = -10
            elif action == 5:  # DROPOFF
                if self.passenger_picked_up:
                    if self.taxi_pos == self.destination:
                        reward += 50
                        return self.get_state(), reward -0.1, True, {}
                    else:
                        reward -=10
                    self.passenger_picked_up = False
                    self.passenger_loc = self.taxi_pos
                else:
                    reward -=10

        reward -= 0.1

        self.current_fuel -= 1
        if self.current_fuel <= 0:
            return self.get_state(), reward -10, True, {}



        return self.get_state(), reward, False, {}

    def get_state(self):
        """Return the current environment state."""
        taxi_row, taxi_col = self.taxi_pos
        passenger_row, passenger_col = self.passenger_loc
        destination_row, destination_col = self.destination

        obstacle_north = int(taxi_row == 0 or (taxi_row-1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row+1, taxi_col) in self.obstacles)
        obstacle_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col+1) in self.obstacles)
        obstacle_west  = int(taxi_col == 0 or (taxi_row , taxi_col-1) in self.obstacles)

        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east  = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west  = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle  = int( (taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle

        destination_loc_north = int( (taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int( (taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east  = int( (taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west  = int( (taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle  = int( (taxi_row, taxi_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle


        state = (taxi_row, taxi_col, self.stations[0][0],self.stations[0][1] ,self.stations[1][0],self.stations[1][1],self.stations[2][0],self.stations[2][1],self.stations[3][0],self.stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
        return state
    def render_env(self, taxi_pos,   action=None, step=None, fuel=None):
        clear_output(wait=True)

        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]

        '''
        # Place passenger
        py, px = passenger_pos
        if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
            grid[py][px] = 'P'
        '''


        grid[0][0]='R'
        grid[0][4]='G'
        grid[4][0]='Y'
        grid[4][4]='B'
        grid[self.passenger_loc[0]][self.passenger_loc[1]] = 'P'
        '''
        # Place destination
        dy, dx = destination_pos
        if 0 <= dx < self.grid_size and 0 <= dy < self.grid_size:
            grid[dy][dx] = 'D'
        '''
        # Place taxi
        ty, tx = taxi_pos
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            # grid[ty][tx] = '🚖'
            grid[ty][tx] = 'T'

        # Print step info
        print(f"\nStep: {step}")
        print(f"Taxi Position: ({tx}, {ty})")
        #print(f"Passenger Position: ({px}, {py}) {'(In Taxi)' if (px, py) == (tx, ty) else ''}")
        #print(f"Destination: ({dx}, {dy})")
        print(f"Fuel Left: {fuel}")
        print(f"Last Action: {self.get_action_name(action)}\n")

        # Print grid
        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        """Returns a human-readable action name."""
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"


def run_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = SimpleTaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    stations = [(0, 0), (0, 4), (4, 0), (4,4)]

    taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

    if render:
        env.render_env((taxi_row, taxi_col),
                       action=None, step=step_count, fuel=env.current_fuel)
        time.sleep(0.5)
    while not done:


        action = student_agent.get_action(obs)

        obs, reward, done, _ = env.step(action)
        # print('obs=',obs)
        total_reward += reward
        step_count += 1

        taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs

        if render:
            env.render_env((taxi_row, taxi_col),
                           action=action, step=step_count, fuel=env.current_fuel)

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

q_table = {}

def train_agent(agent_file, env_config, episodes=5000, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.999):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    rewards_per_episode = []
    epsilon = epsilon_start

    env = SimpleTaxiEnv(**env_config)

    for episode in range(episodes):

        obs, _ = env.reset()
        total_reward = 0
        total_shaped_reward = 0
        done = False

        student_agent.agent.reset()
        state = student_agent.agent.obs_to_state(obs)

        while not done:
            student_agent.agent.init_state_in_q_table(state)

            if np.random.uniform(0, 1) < epsilon:
                action = random.choice([0, 1, 2, 3, 4, 5])
            else:
                action = student_agent.agent.get_action(state)

            student_agent.agent.update_has_passenger(obs, action)

            # s = [[0,0] for _ in range(4)]
            # r, c, s[0][0], s[0][1], s[1][0], s[1][1], s[2][0], s[2][1], s[3][0], s[3][1], _, _, _, _, passenger_look, destination_look = obs

            before_picked_up = env.passenger_picked_up

            obs, reward, done, _ = env.step(action)
            # print('obs=',obs)
            next_state = student_agent.agent.obs_to_state(obs)

            # if env.passenger_picked_up != student_agent.agent.has_passenger and not done:
            #     env.render_env((obs[0], obs[1]),
            #                action=action, step=-1, fuel=env.current_fuel)
            #     print(env.passenger_picked_up, student_agent.agent.has_passenger, action)
            #     print(env.passenger_loc, env.taxi_pos, student_agent.agent.passenger_pos)
            #     exit(1)

            # nr, nc, _, _, _, _, _, _, _, _, _, _, _, _, passenger_look, destination_look = obs

            # shaped rewards
            shaped_reward = 0
            if env.passenger_picked_up and not before_picked_up:
                shaped_reward += 10
            elif not env.passenger_picked_up and before_picked_up:
                shaped_reward -= 20
            if env.passenger_picked_up:
                shaped_reward += 0.05
            if done:
                shaped_reward += 1000
            if (obs[0], obs[1]) in env.obstacles:
                shaped_reward += 5 # 抵銷reward-5

            total_reward += reward
            reward += shaped_reward

            student_agent.agent.init_state_in_q_table(next_state)

            student_agent.agent.update_q_table(state, next_state, action, alpha, reward, gamma)

            state = next_state
            total_shaped_reward += reward

        rewards_per_episode.append(total_reward)

        epsilon = max(epsilon_end, epsilon * decay_rate)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.1f}, Avg Shaped Reward: {total_shaped_reward:.1f}, Epsilon: {epsilon:.3f}")

    student_agent.agent.save_q_table()

if __name__ == "__main__":
    env_config = {
        "grid_size": 5,
        "fuel_limit": 5000
    }

    # train_agent("student_agent.py", env_config, episodes=20000, decay_rate=0.99985)

    agent_score = run_agent("student_agent.py", env_config, render=True)
    print(f"Final Score: {agent_score}")
