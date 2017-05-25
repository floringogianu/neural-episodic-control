# Bitdefender, 2107

from .atari_conv import AtariConv
from .catch_net import CatchNet
from .catch_conv import CatchConv
from .kernel_density_estimate import KernelDensityEstimate

ESTIMATORS = {
    "atari_conv": AtariConv,
    "catch": CatchNet,
    "catch_conv": CatchConv
}


def get_estimator(estimator_name):
    return ESTIMATORS[estimator_name]
