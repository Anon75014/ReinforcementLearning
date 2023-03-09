import os
import numpy as np
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.callbacks import BaseCallback

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the best training reward achieved during training.

    :param check_freq: (int) Frequency at which to save the model.
    :param log_dir: (str) Directory where the model will be saved.
    """
    def __init__(self, check_freq: int, log_dir: str) -> None:
        super().__init__()
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -float("inf")

    def _init_callback(self) -> None:
        if self.check_freq > 0:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.check_freq > 0 and self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Save the best model
                    self.model.save(os.path.join(self.log_dir, 'best_model.zip'))
        return True