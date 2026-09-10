# src/export_onnx.py
import os
import torch
from src.model import MiniResNet

def export_to_onnx(
    weights_path="./artifacts/optimizer_sgd_momentum_best.pt",
    onnx_output_path="./artifacts/cifar10_miniresnet.onnx"
):
    os.makedirs(os.path.dirname(onnx_output_path), exist_ok=True)
    device = torch.device("cpu")
    
    # 1. Initialize architecture and load champion weights
    model = MiniResNet(num_classes=10)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()  # Freeze BatchNorm layers into static scaling factors

    # 2. Dummy input for PyTorch tracing (1 sample, 3 channels, 32x32)
    dummy_input = torch.randn(1, 3, 32, 32, requires_grad=False)

    # 3. Export to ONNX static graph with dynamic batch dimension
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,  # Fuses Conv+BN where applicable
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported successfully to {onnx_output_path}")

if __name__ == "__main__":
    export_to_onnx()