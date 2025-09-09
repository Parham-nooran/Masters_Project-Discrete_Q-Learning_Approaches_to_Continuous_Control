from torch import nn


class BangBangCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_double_q = True
