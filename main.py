import torch
import numpy
import time
from termcolor import colored as clr

import utils
from utils import Preprocessor
from agents import get_agent


def train_agent(cmdl):
    step_cnt = 0
    ep_cnt = 0
    preprocess = Preprocessor(cmdl.env_class).transform

    env = utils.get_new_env(cmdl.env_name)
    agent = get_agent(cmdl.agent.name)(env.action_space, cmdl.agent)
    display_setup(env, cmdl)

    start_time = time.time()
    while step_cnt < cmdl.training.step_no:

        ep_cnt += 1
        o, r, done = env.reset(), 0, False
        s = preprocess(o)

        while not done:
            a = agent.evaluate_policy(s)
            o, r, done, _ = env.step(a)
            _s, _a = s, a
            s = preprocess(o)
            agent.improve_policy(_s, _a, r, s, done)

            step_cnt += 1
            agent.gather_stats(r, done)

        if ep_cnt % cmdl.report_freq == 0:
            agent.display_stats(start_time)
            agent.display_model_stats()

    end_time = time.time()
    display_stats(ep_cnt, step_cnt, end_time - start_time)
    """
    for i in range(env.action_space.n):
        print(agent.kv[i].keys)
    """


def display_setup(env, config):
    print("----------------------------")
    print("Seed         : %d" % config.seed)
    print("DND size     : %d" % config.agent.dnd.size)
    print("DND number   : %d" % env.action_space.n)
    print("Feature size : %s" % str(config.agent.dnd.linear_projection or
          "Conv out."))
    print("K-nearest    : %d" % config.agent.dnd.knn_no)
    print("N-step       : %d" % config.agent.n_horizon)
    print("Optim step   : %d" % config.agent.update_freq)
    print("L-rate       : %.6f" % config.agent.lr)
    print("----------------------------")
    print("stp, nst, act  |  return")
    print("----------------------------")


def display_stats(ep_cnt, step_cnt, elapsed_time):
    fps = step_cnt / elapsed_time
    print(clr("[  %s   ] finished after %d eps, %d steps."
          % ("Main", ep_cnt, step_cnt), 'white', 'on_magenta'))
    print(clr("[  %s   ] finished after %.2fs, %.2ffps."
          % ("Main", elapsed_time, fps), 'white', 'on_magenta'))


if __name__ == "__main__":
    cmd_args = utils.parse_cmd_args()
    config = utils.parse_config_file(cmd_args)
    torch.manual_seed(config.seed)
    numpy.random.seed(config.seed)

    train_agent(config)
