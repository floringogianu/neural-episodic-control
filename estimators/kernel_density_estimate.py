""" Differentiable Kernel Estimate.
"""
import torch


class KernelDensityEstimate(object):
    def __init__(self, delta=0.001):
        self.delta = delta

    def gaussian_kernel(self, x, x_i):
        """ Computes 1 / norm(x - x_i).
        """
        norm = torch.norm(x_i - x.expand_as(x_i), 2, 1)
        return 1 / torch.add(norm, self.delta)

    def normalized_kernel(self, distances):
        """ Computes k(x - x_i) / sum(k(x - x_i)).
        """
        if isinstance(distances, torch.autograd.Variable):
            return distances / distances.sum().expand_as(distances)
        elif isinstance(distances, torch.Tensor):
            return distances / distances.sum()
