from .nec_agent import NECAgent
from .dqn_agent import DQNAgent
from .random_agent import RandomAgent

AGENTS = {
    "nec": NECAgent,
    "dqn": DQNAgent,
    "random": RandomAgent
}


def get_agent(name):
    return AGENTS[name]
