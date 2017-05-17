# Bitdefender, 2107

from estimators.atari_net import AtariNet
from estimators.mini_net import MiniNet
from estimators.catch_net import CatchNet
from estimators.catch_conv import CatchConv
from estimators.dnd import DND
from estimators.kernel_density_estimate import KernelDensityEstimate

ESTIMATORS = {
    "atari": AtariNet,
    "mini": MiniNet,
    "catch": CatchNet,
    "catch_conv": CatchConv
}


def get_estimator(estimator_name):
    return ESTIMATORS[estimator_name]
