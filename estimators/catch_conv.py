""" Neural Network architecture for low-dimensional games.
"""

import torch.nn as nn
import torch.nn.functional as F


def conv_out_dim(w, conv):
    k = conv.kernel_size[0]
    s = conv.stride[0]
    p = conv.padding[0]
    return int((w - k + 2 * p) / s + 1)


class CatchConv(nn.Module):
    def __init__(self, state_dim=(1, 24), linear_projection=None):
        super(CatchConv, self).__init__()

        self.in_channels, self.in_width = state_dim

        self.conv1 = nn.Conv2d(self.in_channels, 32, 5, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)

        map_width1 = conv_out_dim(self.in_width, self.conv1)
        map_width2 = conv_out_dim(map_width1, self.conv2)
        if linear_projection:
            self.embedding_size = linear_projection

            self.lin1 = nn.Linear(32 * map_width2**2, linear_projection)
            self.forward = self._linear_projection
        else:
            self.forward = self._no_projection
            self.embedding_size = 32 * map_width2**2

    def _no_projection(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x.view(x.size(0), -1)

    def _linear_projection(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return F.relu(self.lin1(x.view(x.size(0), -1)))

    def get_embedding_size(self):
        return self.embedding_size


if __name__ == "__main__":
    import torch
    from torch.autograd import Variable

    model = CatchConv(state_dim=(1, 24))
    print(model)
    print(model(Variable(torch.rand(1, 1, 24, 24))).size())

    model = CatchConv(state_dim=(1, 24), linear_projection=128)
    print(model)
    print(model(Variable(torch.rand(1, 1, 24, 24))).size())
