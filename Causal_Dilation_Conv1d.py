import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CausalDilationConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalDilationConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalDilationConv1d, self).forward(F.pad(input, (self.__padding, 0)))
if __name__ == '__main__':
    input = torch.from_numpy(np.ones((3, 2, 5))).float()
    CaConv1d = CausalDilationConv1d(in_channels=2, out_channels=6, kernel_size=2, dilation=1)
    out = CaConv1d(input)
    print(out.size())
