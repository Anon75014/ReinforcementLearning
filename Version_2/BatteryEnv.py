import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class BatteryEnv(gym.Env):
    """A battery trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data):
        super(BatteryEnv, self).__init__()

        self.data = data  # Price data from the Entso-e platform
        self.max_price = np.max(data)  # Maximum price for normalization
        self.reward_range = (-np.inf, np.inf)
        self.current_step = 0
        self.soc = 0.5  # Battery starts at SoC 50
        self.episode_over = False  # End of episode flag
        self.profit = 0  # Profit at current time step
        self.reward = 0
        self.profit_history = []  # Profit stored over an episode
        self.profit_episode = []  # For use in the render() method

        # Actions of the format Charge x%, Discharge x%, Hold, etc.
        self.action_space = spaces.Discrete(3)

        # Observations contain the day-ahead price values for the next 24 hours, as well as SoC and Profit
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(26,), dtype=np.float32)

    def _next_observation(self):
        # Get the day-ahead price data points for the next 24 hours and scale to between 0-1
        # Also get the current SoC and profit
        obs = np.concatenate([
            self.data.flatten() / self.max_price,
            np.array([self.soc, self.profit / self.max_price])
        ], dtype=np.float32)

        return obs

    def _take_action(self, action):
        # Set the current price to the day-ahead price at the current time step
        current_price = self.data[self.current_step]

        if action == 0 and self.soc < 1:
            # Charge the battery by 25%
            self.profit -= current_price
            self.soc += 0.25


        elif action == 1 and self.soc > 0:
            # Discharge the battery by 25%
            self.profit += current_price
            self.soc -= 0.25


        else:   # Hold
            pass


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1
        self.profit_history.append(self.profit)

        # End of episode
        if self.current_step == 24:
            self.reward = self.profit
            self.episode_over = True  # Raises the end of the episode flag
            self.profit_episode = self.profit_history.copy()  # Store episode's profit history for render()
            self.profit_history = []  # Reset profit history for next episode
        else:
            self.episode_over = False

        # Perform next observation and keep track of the profit
        obs = self._next_observation()

        return obs, self.reward, self.episode_over, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.soc = 0.5  # Set initial state of charge to 50%
        self.profit = 0  # Reset profits for next episode
        self.reward = 0

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Plot profit history over an episode
        if self.episode_over:
            plt.plot(self.profit_episode)
            plt.xlabel('Time Step')
            plt.ylabel('Profit')
            plt.title('Profit as a function of Time Step')
            plt.show()

