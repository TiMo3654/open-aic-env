from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):

    def __init__(self, env, verbose=0,):
        super().__init__(verbose)

        self.env    = env

    def _on_training_start(self):
        self._log_freq = 1  # log every 10 calls

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        x = self.env.logger_values['x']
        y = self.env.logger_values['y']
        idx = self.env.logger_values['idx']
        reward = self.env.logger_values['reward']
        #wl = self.env.logger_values['wl']


        self.logger.record("env/x", x)
        self.logger.record("env/y", y)
        self.logger.record("env/idx", idx)
        self.logger.record("env/reward", reward)
        #self.logger.record("env/WL", wl)

        return True