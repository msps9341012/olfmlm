import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MatrixReconstruction(nn.Module):
    def __init__(self, batch_size, nbow, ntopic, device):
        super(MatrixReconstruction, self).__init__()
        self.coeff = nn.Parameter(torch.randn(batch_size, nbow, ntopic, device=device, requires_grad=True))
        self.device = device

    def compute_coeff_pos(self):
        self.coeff.data = self.coeff.clamp(0.0, 1.0)

    def forward(self, input):
        result = self.coeff.matmul(input)
        return result

