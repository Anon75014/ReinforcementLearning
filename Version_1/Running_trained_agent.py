import gym
from gym import spaces
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from Creating_env import BatteryEnv


# Create an instance of the BatteryEnv class
env_test = BatteryEnv()
# Wrap the environment with a Monitor wrapper
eval_env_2 = Monitor(env_test)
# Load the trained agent
new_model = PPO.load('trained_PPO_agent_24_observations', env=env_test)

# Use the trained agent on the new environment
mean_reward, _ = evaluate_policy(new_model, eval_env_2, n_eval_episodes=10)
print('Mean reward:', mean_reward)


def agent_run_evaluation(model, env):
    obs = env.reset()
    done = False
    time_step = 0
    profits_history = [0]

    while time_step < 25:
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