""" Neural Network architecture for Atari games.
"""

import torch.nn as nn
import torch.nn.functional as F


def conv_out_dim(w, conv):
    k = conv.kernel_size[0]
    s = conv.stride[0]
    p = conv.padding[0]
    return int((w - k + 2 * p) / s + 1)


class AtariConv(nn.Module):
    def __init__(self, state_dim=(1, 24), linear_projection=None):
        super(AtariConv, self).__init__()
        self.in_channels, self.in_width = state_dim

        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        map_width1 = conv_out_dim(self.in_width, self.conv1)
        map_width2 = conv_out_dim(map_width1, self.conv2)
        map_width3 = conv_out_dim(map_width2, self.conv3)
        if linear_projection:
            self.embedding_size = linear_projection
            self.lin1 = nn.Linear(64 * map_width3**2, linear_projection)
            self.forward = self._linear_projection
        else:
            self.forward = self._no_projection
            self.embedding_size = 64 * map_width3**2

    def _no_projection(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1)

    def _linear_projection(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return F.relu(self.lin1(x.view(x.size(0), -1)))

    def get_embedding_size(self):
        return self.embedding_size


if __name__ == "__main__":
    import torch
    from torch.autograd import Variable

    model = AtariConv(state_dim=(1, 84))
    print(model)
    print(model(Variable(torch.rand(1, 1, 84, 84))).size())

    model = AtariConv(state_dim=(1, 84), linear_projection=128)
    print(model)
    print(model(Variable(torch.rand(1, 1, 84, 84))).size())
