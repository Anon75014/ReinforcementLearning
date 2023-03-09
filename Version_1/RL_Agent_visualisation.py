import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import time

class BatteryEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, prices_day_1):
        super(BatteryEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # Charge, Discharge, Neutral
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # Initialize the environment
        self.prices_day_1 = prices_day_1
        self.SoC = 50
        self.profits = 0
        self.time_step = 0
        self.action = None  # Initialize the action attribute


    def step(self, action):
        # Set the action attribute of the environment
        self.action = action
        # Compute new battery state of charge based on battery behavior
        if action == 1 and self.SoC - 25 >= 0:  # Discharge
            self.SoC -= 25
            self.profits += self.prices_day_1[self.time_step+1]
        elif action == 0 and self.SoC + 25 <= 100:  # Charge
            self.SoC += 25
            self.profits -= self.prices_day_1[self.time_step+1]
        else:  # Neutral
            pass

        # Update the state of the environment
        self.time_step += 1
        #self.profits_history.append(self.profits)  # Add the current profits to the history
        done = self.time_step == len(self.prices_day_1) - 1
        reward = self.profits if done else 0
        observation = np.array([self.prices_day_1[self.time_step], self.SoC, self.profits], dtype=np.float32)


        return observation, reward, done, {}

    def reset(self):
        # Reset the environment
        self.SoC = 50
        self.profits = 0
        self.time_step = 0
        observation = np.array([self.prices_day_1[self.time_step], self.SoC, self.profits], dtype=np.float32)

        return observation

    def render(self, mode='human'):
        pass

    def close(self):
        # Close the environment (optional)
        pass

'''
# Load the prices for day 1
prices_day_1 = np.loadtxt('France_price_2022_F.csv', delimiter=',', usecols=1, skiprows=0, max_rows=24)
# Create an instance of the BatteryEnv class
env = BatteryEnv(prices_day_1)
# Create a vectorized environment
env = make_vec_env(lambda: env, n_envs=1)
# Instantiate a PPO agent
model = PPO('MlpPolicy', env, verbose=1)
# Train the agent for N time steps
time_steps = 20000
model.learn(total_timesteps=time_steps)
# Evaluate the agent
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
print('Mean reward:', mean_reward)
# Save the trained agent
model.save('trained_agent_light')
'''
# Load the prices for a given day
day_number = 1
prices_day = np.loadtxt('France_price_2022_F.csv', delimiter=',', usecols=1, skiprows=(day_number-1)*24, max_rows=24)
# Create an instance of the BatteryEnv class
env_test = BatteryEnv(prices_day)
# Load the trained agent
new_model = PPO.load('trained_agent_light', env=env_test)

# Use the trained agent on the new environment
mean_reward, _ = evaluate_policy(new_model, env_test, n_eval_episodes=10)
print('Mean reward:', mean_reward)

def loop_agent(new_model, env_test, n_episodes):
    rewards = []
    for i in range(n_episodes):
        obs = env_test.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = new_model.predict(obs)
            obs, reward, done, _ = env_test.step(action)
            total_reward += reward
        rewards.append(total_reward)
        print('Episode', i+1, 'reward:', total_reward)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per episode')
    plt.show()

# Loop the agent over multiple episodes and plot the rewards
loop_agent(new_model, env_test, n_episodes=15)

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
plt.xlabel('Time steps')
plt.ylabel('Profits')
plt.title('Profits over time')
plt.show()






