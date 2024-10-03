import torch
import torch.nn as nn
import torch.onnx
from stable_baselines3 import SAC
import numpy as np
from data_loader import *
from imports import *
from functions import *
import warnings
warnings.filterwarnings("ignore")

class DeterministicSACPolicy(nn.Module):
    def __init__(self, sb3_policy):
        super(DeterministicSACPolicy, self).__init__()
        self.actor = sb3_policy.actor
        self.action_space = sb3_policy.action_space

    def forward(self, obs):
        # Get mean actions from actor
        mean_actions = self.actor(obs)
        scaled_actions = torch.tanh(mean_actions)
        # Scale actions in the action's space scope
        low = torch.tensor(self.action_space.low, dtype=scaled_actions.dtype, device=scaled_actions.device)
        high = torch.tensor(self.action_space.high, dtype=scaled_actions.dtype, device=scaled_actions.device)
        actions = low + 0.5 * (scaled_actions + 1.0) * (high - low)
        return actions

model_path = 'models/1200ep/OldMedWeek_WorstB_1200/sac_citylearn_final.zip'
model = SAC.load(model_path)

# Deterministic policy network
deterministic_policy = DeterministicSACPolicy(model.policy)
deterministic_policy.eval()

# Dummy input for input observation 
obs_size = model.observation_space.shape[0]
dummy_input = torch.randn(1, obs_size, dtype=torch.float32)

# Export model into ONNX
torch.onnx.export(
    deterministic_policy,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    #dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} #Won't be Suported from MCU with Dynamic inputs
)
