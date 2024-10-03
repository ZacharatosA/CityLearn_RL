from imports import*
import torch
import torch.onnx
from onnx import numpy_helper
import onnx

#Settings
ei.API_KEY = "your_presonal_ei_api_key"
# devices=ei.model.list_profile_devices()
# print(f"Available devices: {devices}")


# Profile model
try:
    profile = ei.model.profile(model='model.onnx')
    print(profile.summary())
except Exception as e:
    print(f"Could not profile: {e}")


