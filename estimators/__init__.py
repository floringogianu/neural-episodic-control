# Bitdefender, 2107

from .atari_net import AtariNet
from .mini_net import MiniNet
from .catch_net import CatchNet
from .catch_conv import CatchConv
from .kernel_density_estimate import KernelDensityEstimate

ESTIMATORS = {
    "atari": AtariNet,
    "mini": MiniNet,
    "catch": CatchNet,
    "catch_conv": CatchConv
}


def get_estimator(estimator_name):
    return ESTIMATORS[estimator_name]
