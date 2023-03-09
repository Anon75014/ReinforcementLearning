import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

class BatteryEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, prices):
        super(BatteryEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # Charge, Discharge, Neutral
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # Initialize the environment
        n = len(prices)
        x = random.randint(0,n/24-1) #pick a random day in prices (0 if prices is the length of a day)
        self.prices_day = prices[x*24:(x+1)*24] #select the prices corresponding to this day
        self.SoC = 50 #initial condition for SoC
        self.profits = 0
        self.time_step = 0
        self.profits_history = [0]


    def step(self, action):
        # Compute new battery state of charge based on battery behavior
        if action == 1 and self.SoC - 25 >= 0:  # Discharge
            self.SoC -= 25
            self.profits += self.prices_day[self.time_step]
        elif action == 0 and self.SoC + 25 <= 100:  # Charge
            self.SoC += 25
            self.profits -= self.prices_day[self.time_step]
        else:  # Neutral
            pass

        # Update the state of the environment
        self.time_step += 1
        self.profits_history.append(self.profits)  # Add the current profits to the history
        done = self.time_step == len(self.prices_day)-1
        reward = self.profits if done else 0
        observation = np.array([self.prices_day[self.time_step], self.SoC, self.profits], dtype=np.float32)


        return observation, reward, done, {}

    def reset(self):
        # Reset the environment
        n = len(prices)
        x = random.randint(0,n/24-1) #pick a random day in the price array
        self.prices_day = prices[x*24:(x+1)*24] #select the prices corresponding to this day
        self.SoC = 50
        self.profits = 0
        self.time_step = 0
        observation = np.array([self.prices_day[self.time_step], self.SoC, self.profits], dtype=np.float32)

        return observation

    def render(self, mode='human'):
        pass

    def close(self):
        # Close the environment (optional)
        pass


# Load the prices
prices = np.loadtxt('France_price_2022_F.csv', delimiter=',', usecols=1, max_rows=8760)
# Create an instance of the BatteryEnv class
env = BatteryEnv(prices)
# Create a vectorized environment
env = make_vec_env(lambda: env, n_envs=1)
# Instantiate a PPO agent
model = PPO('MlpPolicy', env, verbose=1)
# Train the agent for N time steps
time_steps = 10000
model.learn(total_timesteps=time_steps)
# Evaluate the agent
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
print('Mean reward:', mean_reward)
# Save the trained agent
model.save('trained_agent')


# Load the prices for day x
x=0
prices = np.loadtxt('France_price_2022_F.csv', delimiter=',', usecols=1, skiprows=x*24, max_rows=24)
# Create an instance of the BatteryEnv class
env_test = BatteryEnv(prices)
# Wrap the environment with a Monitor wrapper
eval_env_2 = Monitor(env_test)
# Load the trained agent
new_model = PPO.load('trained_agent', env=env_test)

# Use the trained agent on the new environment
mean_reward, _ = evaluate_policy(new_model, eval_env_2, n_eval_episodes=10)
print('Mean reward:', mean_reward)


def agent_run_evaluation(model, env):
    obs = env.reset()
    done = False
    time_step = 0
    profits_history = []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        time_step += 1
        profits_history.append(env.profits)

    return profits_history


profits_history = agent_run_evaluation(new_model, env_test)
plt.plot(profits_history)
plt.xlabel('Time steps (h)')
plt.ylabel('Profits (€)')
plt.title('Profits over time')
plt.show()







