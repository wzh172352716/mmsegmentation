from mmseg.apis import init_model, inference_model, show_result_pyplot
import numpy as np

# 初始化模型
config_path = '../configs/deeplabv3plus/deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024.py'
checkpoint_path = '../checkpoint/deeplabv3_r18-d8_512x1024_80k_cityscapes_20201225_021506-23dffbe2.pth'
# checkpoint_file = './checkpoint/deeplabv3_r18-d8_512x1024_80k_cityscapes_20201225_021506-23dffbe2.pth'
img_path = '../data/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000576_leftImg8bit.png'

# 对图像进行推理
model = init_model(config_path, checkpoint_path, device='cpu')
result = inference_model(model, img_path)
vis_iamge = show_result_pyplot(model, img_path, result, save_dir='results', out_file='results/result.png')