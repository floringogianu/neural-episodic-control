import torch
import time
from termcolor import colored as clr
from utils import not_implemented


class BaseAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.action_no = self.action_space.n

        self.step_cnt = 0
        self.ep_cnt = 0
        self.ep_reward_cnt = 0
        self.ep_reward = []

    def evaluate_policy(self, obs):
        not_implemented(self)

    def improve_policy(self, _state, _action, reward, state, done):
        not_implemented(self)

    def gather_stats(self, reward, done):
        self.step_cnt += 1
        self.ep_reward_cnt += reward
        if done:
            self.ep_cnt += 1
            self.ep_reward.append(self.ep_reward_cnt)
            self.ep_reward_cnt = 0

    def display_stats(self, start_time):
        mean_rw = torch.FloatTensor([self.ep_reward]).mean()
        fps = self.step_cnt / (time.time() - start_time)
        print(clr("[%s] ep=%6d, reward/ep=%3.2f, fps=%.2f"
              % (self.name, len(self.ep_reward), mean_rw, fps),
              'white', 'on_cyan'))
        self.ep_reward.clear()

    def display_model_stats(self):
        pass
