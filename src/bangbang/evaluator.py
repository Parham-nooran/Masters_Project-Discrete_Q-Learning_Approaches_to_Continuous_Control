import torch
import numpy as np
import matplotlib.pyplot as plt
from dm_control import suite
import time
import os

from agent import BangBangAgent
from config import create_bangbang_config_from_args
from train import process_observation, get_obs_shape, load_bangbang_checkpoint

class BangBangEvaluator:
    def __init__(self, agent, env, device="cuda" if ):
        self.agent = agent
