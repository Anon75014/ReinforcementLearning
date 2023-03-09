import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from BatteryEnv import BatteryEnv


data = np.loadtxt('France_price_2022.csv', delimiter=',', usecols=[1], skiprows=0, max_rows=24)

# The algorithms require a vectorized environment to run
env = make_vec_env(lambda: BatteryEnv(data), n_envs=1)

# Train the agent and save the trained model to a file
model = PPO("MlpPolicy", env=env, verbose=1)
model.learn(total_timesteps=10000)
model.save("battery_trading_model")

# Load the saved model and run the agent
model = PPO.load("battery_trading_model")
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
print('Mean reward:', mean_reward)
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()

# Close the environment
env.close()