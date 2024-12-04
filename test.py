import torch
from model import DDAF
from modules import SEANetEncoder, SEANetDecoder

encoder = SEANetEncoder()
decoder = SEANetDecoder()

model = DDAF(encoder, decoder, d_model=128, num_classes=2, fusion_depth=8)
print(model)
# for key in checkpoint:
#     print(key)
#     print(checkpoint[key].shape)
#     # print(checkpoint[key])
#     # print()
