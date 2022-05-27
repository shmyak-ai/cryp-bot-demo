import random

import numpy as np
import gym
from gym import spaces
from ray.rllib.env.env_context import EnvContext


def prepare_dict(df):
    price_array = df['price'].to_numpy(dtype=np.float32)[:, np.newaxis]
    df = df.drop(columns=['price'])
    obs_array = df.to_numpy(dtype=np.float32)
    data_dictionary = {'price_array': price_array, 'observations': obs_array}
    return data_dictionary


class CryptoEnv(gym.Env):
    def __init__(self, config: EnvContext):
        self._price_array = config['price_array']
        self._observations = config['observations']

        self._base_cash = config['initial_capital']
        self._cash_usd = None
        self._stocks_usd = None
        self._stocks = None
        self._total_asset = None
        self._initial_total_asset = None

        self._time_step = None
        self._initial_step = None
        self._max_steps = config['max_steps']
        self._final_step = None
        self._upper_bound_step = self._price_array.shape[0] - self._max_steps - 1

        self._gamma = config['gamma']
        self._gamma_return = None

        self._action_dim = self._price_array.shape[1]
        # buy or sell up to the base cash equivalent(usd)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self._action_dim,), dtype=np.float32)
        # cash + stocks + observations
        self._state_dim = 2 + self._observations.shape[1]
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(self._state_dim,), dtype=np.float32)

        self._state = None
        self._episode_ended = None

    def reset(self):
        self._time_step = self._initial_step = random.randint(0, self._upper_bound_step)
        self._final_step = self._initial_step + self._max_steps
        self._cash_usd = random.random() * self._base_cash
        self._stocks_usd = random.random() * self._base_cash
        self._stocks = self._stocks_usd / self._price_array[self._time_step][0]
        self._total_asset = self._initial_total_asset = self._cash_usd + self._stocks_usd
        self._gamma_return = 0.0

        self._state = self._get_state()
        self._episode_ended = False
        return self._state

    def step(self, action):
        self._time_step += 1
        price = self._price_array[self._time_step][0]
        self._stocks_usd = self._stocks * price
        if action[0] < 0 and price > 0:  # sell
            sell_shares_usd = min(self._base_cash * -action[0], self._stocks_usd)
            self._stocks_usd -= sell_shares_usd
            self._cash_usd += sell_shares_usd
        elif action[0] > 0 and price > 0:  # buy
            money_to_spend = min(self._base_cash * action[0], self._cash_usd)
            self._stocks_usd += money_to_spend
            self._cash_usd -= money_to_spend
        self._stocks = self._stocks_usd / price

        self._episode_ended = self._time_step == self._final_step
        self._state = self._get_state()
        next_total_asset = self._cash_usd + self._stocks_usd

        reward = (next_total_asset - self._total_asset) / self._base_cash
        self._total_asset = next_total_asset
        self._gamma_return = self._gamma_return * self._gamma + reward
        if self._episode_ended:
            reward = self._gamma_return
            return self._state, reward, True, self._get_info()
        else:
            return self._state, reward, False, self._get_info()

    def _get_state(self):
        state = np.hstack(((self._cash_usd - self._base_cash) / self._base_cash, 
                           (self._stocks_usd - self._base_cash) / self._base_cash))
        observation = self._observations[self._time_step]
        state = np.hstack((state, observation)).astype(np.float32)
        return state

    def _get_info(self):
        return {"Initial step": self._initial_step, 
                "Final step": self._final_step, 
                "Initial total asset": self._initial_total_asset, 
                "Final total asset": self._total_asset, 
                "Gamma return": self._gamma_return}
