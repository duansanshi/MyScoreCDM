import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from einops import rearrange, repeat
from omegaconf import DictConfig
import opt_einsum as oe
import numpy as np
from IPython import embed
from functools import partial
from torch.distributions import Normal
from ScConv import  *

contract = oe.contract

L = 24

b = torch.linspace(0, 0, 24)
for i in range(2):
    a = torch.cos(torch.linspace(-1 * i * math.pi, i * math.pi, 24))
    b = b + a

c = b.clone().detach().unsqueeze(-1).repeat(1,3).cuda(3)
print(c.shape)

a = torch.ones(24,3).cuda(3)

k_f = torch.fft.rfft(c, n=2*L, dim=0)  # (C H L)

u_f = torch.fft.rfft(a, n=2*L, dim=0)  # (B H L)

y_f = contract('cl,cl->cl', u_f, k_f)

y = torch.fft.irfft(y_f, n=2*24)[..., :24]  # (B C H L)

print(y)