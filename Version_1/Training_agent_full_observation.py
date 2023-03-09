import matplotlib.pyplot as plt
import pickle
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import DQN
from stable_baselines3.common import results_plotter
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from Creating_env import BatteryEnv
from callbacks import SaveOnBestTrainingRewardCallback
import numpy as np

# Create an instance of the BatteryEnv class
env = BatteryEnv()

# Create a vectorized environment
env = make_vec_env(lambda: env, n_envs=1)
# Instantiate an agent
model = PPO('MlpPolicy', env, verbose=1)
# Define a callback to save the best model
#callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir='./logs')
# Train the agent for N time steps
time_steps = 6e4
model.learn(total_timesteps=time_steps)
obs = env.reset()
# Evaluate the agent
mean_reward2, _ = evaluate_policy(model, env, n_eval_episodes=10)
print('Mean reward:', mean_reward2)
# Save the trained agent
model.save('trained_PPO_agent_24_observations')

# Plot the results
#results_plotter.plot_results(["C:/Users/33631/PycharmProjects/ReinforcementLearning/logs"], 10e6,results_plotter.X_TIMESTEPS, "BatteryEnv")
#plt.show()



