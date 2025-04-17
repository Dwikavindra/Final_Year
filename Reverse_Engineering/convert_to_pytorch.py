import os
import torch
import onnx
import tflite2onnx
from onnx2torch import convert

input_dir = "ensemble-15-mnist"
onnx_dir = "temp_onnx"
output_dir = "ensemble-15-mnist-pytorch"

os.makedirs(onnx_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

for i in range(15):
    model_name = f"ensemble_model_{i}"
    tflite_path = os.path.join(input_dir, f"{model_name}.tflite")
    onnx_path = os.path.join(onnx_dir, f"{model_name}.onnx")
    pytorch_path = os.path.join(output_dir, f"{model_name}.pth")

    
    tflite2onnx.convert(tflite_path=tflite_path, onnx_path=onnx_path)

    
    onnx_model = onnx.load(onnx_path)
    pytorch_model = convert(onnx_model)
    pytorch_model.eval()

    torch.save(pytorch_model, pytorch_path)
    print(f"Saved PyTorch model to {pytorch_path}\n")

print("All ensemble models converted to PyTorch.")
