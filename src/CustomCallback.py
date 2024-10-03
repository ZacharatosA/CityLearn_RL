from imports import*

class CustomCallback(BaseCallback):
    def __init__(self, env: CityLearnEnv, loader: tqdm, trial=None, pruning_freq=100, episodes_to_check=100):
        """Initialize CustomCallback.

        Parameters
        ----------
        env: CityLearnEnv
            CityLearn environment instance.
        loader: tqdm
            Progress bar.
        trial: optuna.trial.Trial
            Optuna trial object.
        check_freq: int
            Frequency (in episodes) to check for pruning.
        """
        super().__init__(verbose=0)
        self.loader = loader
        self.env = env
        self.trial = trial
        self.Pruning_freq = pruning_freq  # Pruning check frequency
        self.episodes_to_check = episodes_to_check
        self.reward_history = []
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        """Called each time the env step function is called."""

        # Update the loader
        self.loader.update(1) 

        # Get the reward of the step
        reward = self.locals['rewards'][0]
        self.episode_rewards.append(reward)

        # Check if the episode has ended
        done = self.locals['dones'][0]
        if done:
            # The episode has ended
            total_episode_reward = sum(self.episode_rewards)
            self.reward_history.append(total_episode_reward)
            self.episode_rewards = []
            self.episode_count += 1

            # Every Pruning_freq episodes, report to Optuna
            if self.episode_count % self.Pruning_freq == 0:
                # Calculate average reward over the last M episodes
                M = min(self.episodes_to_check, len(self.reward_history))  # We use the last episodes_to_check episodes
                if M > 0:
                    intermediate_value = np.mean(self.reward_history[-M:])
                    if self.trial is not None:
                        # Report the intermediate metric to Optuna
                        self.trial.report(intermediate_value, step=self.episode_count)
                        # Check if the trial should be pruned
                        if self.trial.should_prune():
                            raise optuna.exceptions.TrialPruned()

        return True
