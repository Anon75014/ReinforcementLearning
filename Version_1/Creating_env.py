import gym
from gym import spaces
import numpy as np
import random
import pickle



class BatteryEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(BatteryEnv, self).__init__()

        # Load the prices
        self.prices_day = np.loadtxt('France_price_2022_F.csv', delimiter=',', usecols=1, skiprows=0,
                                 max_rows=24)  # max:8760

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # Charge, Discharge, Neutral
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)

        # Initialize the environment
        self.max_price = max(self.prices_day)  # computes the extreme prices for the day
        self.min_price = min(self.prices_day)
        self.SoC = 50  # initial condition for SoC
        self.profits = 0
        self.time_step = 0
        self.profits_history = [0]
        # reward values
        self.penalty_for_invalid_action = -10  # small penalty for wrong actions (eg charging at SoC 100)
        self.reward_for_good_trade = 80
        self.penalty_for_bad_trade = -40



    def step(self, action):
        reward = 0
        # Compute new battery state of charge based on battery behavior
        if action == 1 and self.SoC > 0:  # Discharge
            self.SoC -= 25
            self.profits += self.prices_day[self.time_step]
            self.time_step += 1
        elif action == 0 and self.SoC < 100:  # Charge
            self.SoC += 25
            self.profits -= self.prices_day[self.time_step]
            self.time_step += 1
        elif action == 2:
            self.time_step += 1
        else:  # Invalid action
            reward += self.penalty_for_invalid_action

        # Update the state of the environment

        self.profits_history.append(self.profits)  # Add the current profits to the history
        done = self.time_step == 24
        if done:
            reward = self.profits
        else:
            if action == 0 and self.min_price <= self.prices_day[self.time_step] <= 1.2 * self.min_price:  # Reward for charging when electricity is cheap
                reward += self.reward_for_good_trade
            elif action == 1 and self.min_price <= self.prices_day[self.time_step] <= 1.2 * self.min_price:  # Penalty for discharging when electricity is cheap
                reward += self.penalty_for_bad_trade
            elif action == 1 and 0.8 * self.max_price <= self.prices_day[self.time_step] <= self.max_price:  # Reward for discharging when electricity is expensive
                reward += self.reward_for_good_trade
            elif action == 0 and 0.8 * self.max_price <= self.prices_day[self.time_step] <= self.max_price:  # Penalty for charging when electricity is expensive
                reward += self.penalty_for_bad_trade
        observation = np.concatenate(
            (self.prices_day, np.array([self.SoC, self.profits])), axis=0
        ).astype(np.float32)


        return observation, reward, done, {}

    def reset(self):
        # Reset the environment
        self.SoC = 50
        self.profits = 0
        self.time_step = 0
        observation = np.concatenate(
            (self.prices_day, np.array([self.SoC, self.profits])), axis=0
        ).astype(np.float32)

        return observation

    def render(self, mode='human'):
        pass

    def close(self):
        # Close the environment (optional)
        pass



