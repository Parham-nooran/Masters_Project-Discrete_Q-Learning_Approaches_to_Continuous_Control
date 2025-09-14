import torch


class BangBangEvaluator:
    def __init__(
        self,
        agent,
        env,
        device="cuda" if torch.cuda.is_available() else torch.device("cpu"),
    ):
        self.agent = agent
