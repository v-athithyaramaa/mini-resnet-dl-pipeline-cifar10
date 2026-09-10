# src/export_assignment2.py
import os
import json
import torch
import numpy as np
from src.model import MiniResNet

def save_bin(path: str, tensor: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Ensure contiguous float32 little-endian format
    tensor.astype(np.float32).tofile(path)
    print(f"  [+] Dumped: {path} | shape: {tensor.shape} | size: {tensor.nbytes} bytes")

def export_all():
    print("=" * 70)
    print("Exporting Artifacts for Assignment 2 (Bare-Metal C++ Inference)...")
    print("=" * 70)

    # Base destination matching assignment directory structure
    base_dir = "./cpp_engine"
    data_dir = os.path.join(base_dir, "data")
    weights_dir = os.path.join(data_dir, "weights")
    ref_dir = os.path.join(data_dir, "reference")
    config_dir = os.path.join(base_dir, "configs")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    # Load model
    model = MiniResNet(num_classes=10)
    pt_path = "./artifacts/optimizer_sgd_momentum_best.pt"
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Model checkpoint not found at {pt_path}")
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()

    # Deterministic input tensor
    torch.manual_seed(42)
    sample_input = torch.randn(1, 3, 32, 32, dtype=torch.float32)
    input_path = os.path.join(data_dir, "input", "sample_input.bin")
    save_bin(input_path, sample_input.numpy())

    # Layer-by-layer forward execution and reference extraction
    print("\nExecuting forward pass and capturing layer-wise references...")
    with torch.no_grad():
        # 1. Stem
        stem_conv = model.stem[0](sample_input)
        stem_bn = model.stem[1](stem_conv)
        stem_out = model.stem[2](stem_bn) # ReLU

        # 2. Downsample 1
        ds1_conv = model.downsample1[0](stem_out)
        ds1_bn = model.downsample1[1](ds1_conv)
        ds1_out = model.downsample1[2](ds1_bn) # ReLU

        # 3. Residual Block
        res_conv1 = model.res_block.conv1(ds1_out)
        res_bn1 = model.res_block.bn1(res_conv1)
        res_relu1 = model.res_block.relu(res_bn1)
        res_conv2 = model.res_block.conv2(res_relu1)
        res_bn2 = model.res_block.bn2(res_conv2)
        res_add = res_bn2 + ds1_out # Residual identity highway
        res_out = model.res_block.relu(res_add)

        # 4. Downsample 2
        ds2_conv = model.downsample2[0](res_out)
        ds2_bn = model.downsample2[1](ds2_conv)
        ds2_out = model.downsample2[2](ds2_bn) # ReLU

        # 5. Global Average Pooling & Classifier
        gap_out = model.gap(ds2_out)
        flat_out = model.flatten(gap_out)
        logits = model.classifier(flat_out)
        probabilities = torch.softmax(logits, dim=-1)

    # Save all reference outputs
    refs = {
        "ref_stem_conv": stem_conv,
        "ref_stem_bn": stem_bn,
        "ref_stem_out": stem_out,
        "ref_ds1_conv": ds1_conv,
        "ref_ds1_bn": ds1_bn,
        "ref_ds1_out": ds1_out,
        "ref_res_conv1": res_conv1,
        "ref_res_bn1": res_bn1,
        "ref_res_relu1": res_relu1,
        "ref_res_conv2": res_conv2,
        "ref_res_bn2": res_bn2,
        "ref_res_add": res_add,
        "ref_res_out": res_out,
        "ref_ds2_conv": ds2_conv,
        "ref_ds2_bn": ds2_bn,
        "ref_ds2_out": ds2_out,
        "ref_gap_out": gap_out,
        "ref_flat_out": flat_out,
        "ref_logits": logits,
        "ref_probabilities": probabilities
    }
    for name, tensor in refs.items():
        save_bin(os.path.join(ref_dir, f"{name}.bin"), tensor.numpy())

    # Save layer weights and biases
    print("\nSaving model weights and parameters...")
    def dump_conv(prefix, conv_layer):
        save_bin(os.path.join(weights_dir, f"{prefix}_weight.bin"), conv_layer.weight.data.numpy())
        if conv_layer.bias is not None:
            save_bin(os.path.join(weights_dir, f"{prefix}_bias.bin"), conv_layer.bias.data.numpy())

    def dump_bn(prefix, bn_layer):
        save_bin(os.path.join(weights_dir, f"{prefix}_weight.bin"), bn_layer.weight.data.numpy()) # gamma
        save_bin(os.path.join(weights_dir, f"{prefix}_bias.bin"), bn_layer.bias.data.numpy())     # beta
        save_bin(os.path.join(weights_dir, f"{prefix}_mean.bin"), bn_layer.running_mean.data.numpy())
        save_bin(os.path.join(weights_dir, f"{prefix}_var.bin"), bn_layer.running_var.data.numpy())

    dump_conv("stem_conv", model.stem[0])
    dump_bn("stem_bn", model.stem[1])
    dump_conv("ds1_conv", model.downsample1[0])
    dump_bn("ds1_bn", model.downsample1[1])
    dump_conv("res_conv1", model.res_block.conv1)
    dump_bn("res_bn1", model.res_block.bn1)
    dump_conv("res_conv2", model.res_block.conv2)
    dump_bn("res_bn2", model.res_block.bn2)
    dump_conv("ds2_conv", model.downsample2[0])
    dump_bn("ds2_bn", model.downsample2[1])

    # Classifier (Linear / GEMM)
    save_bin(os.path.join(weights_dir, "classifier_weight.bin"), model.classifier.weight.data.numpy())
    save_bin(os.path.join(weights_dir, "classifier_bias.bin"), model.classifier.bias.data.numpy())

    # Build model_config.json (Section 1 Deliverable)
    config = {
        "model_name": "MiniResNet-CIFAR10",
        "precision": "FP32",
        "input_dimensions": [1, 3, 32, 32],
        "layers": [
            {
                "layer_name": "stem_conv",
                "layer_type": "Conv2D",
                "input_dim": [1, 3, 32, 32],
                "output_dim": [1, 32, 32, 32],
                "attributes": {"kernel_size": 3, "stride": 1, "padding": 1},
                "weight_file": "data/weights/stem_conv_weight.bin",
                "bias_file": None,
                "reference_file": "data/reference/ref_stem_conv.bin"
            },
            {
                "layer_name": "stem_bn",
                "layer_type": "BatchNorm",
                "input_dim": [1, 32, 32, 32],
                "output_dim": [1, 32, 32, 32],
                "attributes": {"num_features": 32, "eps": 1e-5},
                "weight_file": "data/weights/stem_bn_weight.bin",
                "bias_file": "data/weights/stem_bn_bias.bin",
                "mean_file": "data/weights/stem_bn_mean.bin",
                "var_file": "data/weights/stem_bn_var.bin",
                "reference_file": "data/reference/ref_stem_bn.bin"
            },
            {
                "layer_name": "stem_relu",
                "layer_type": "ReLU",
                "input_dim": [1, 32, 32, 32],
                "output_dim": [1, 32, 32, 32],
                "attributes": {},
                "reference_file": "data/reference/ref_stem_out.bin"
            },
            {
                "layer_name": "ds1_conv",
                "layer_type": "Conv2D",
                "input_dim": [1, 32, 32, 32],
                "output_dim": [1, 64, 16, 16],
                "attributes": {"kernel_size": 3, "stride": 2, "padding": 1},
                "weight_file": "data/weights/ds1_conv_weight.bin",
                "bias_file": None,
                "reference_file": "data/reference/ref_ds1_conv.bin"
            },
            {
                "layer_name": "ds1_bn",
                "layer_type": "BatchNorm",
                "input_dim": [1, 64, 16, 16],
                "output_dim": [1, 64, 16, 16],
                "attributes": {"num_features": 64, "eps": 1e-5},
                "weight_file": "data/weights/ds1_bn_weight.bin",
                "bias_file": "data/weights/ds1_bn_bias.bin",
                "mean_file": "data/weights/ds1_bn_mean.bin",
                "var_file": "data/weights/ds1_bn_var.bin",
                "reference_file": "data/reference/ref_ds1_bn.bin"
            },
            {
                "layer_name": "ds1_relu",
                "layer_type": "ReLU",
                "input_dim": [1, 64, 16, 16],
                "output_dim": [1, 64, 16, 16],
                "attributes": {},
                "reference_file": "data/reference/ref_ds1_out.bin"
            },
            {
                "layer_name": "res_conv1",
                "layer_type": "Conv2D",
                "input_dim": [1, 64, 16, 16],
                "output_dim": [1, 64, 16, 16],
                "attributes": {"kernel_size": 3, "stride": 1, "padding": 1},
                "weight_file": "data/weights/res_conv1_weight.bin",
                "bias_file": None,
                "reference_file": "data/reference/ref_res_conv1.bin"
            },
            {
                "layer_name": "res_bn1",
                "layer_type": "BatchNorm",
                "input_dim": [1, 64, 16, 16],
                "output_dim": [1, 64, 16, 16],
                "attributes": {"num_features": 64, "eps": 1e-5},
                "weight_file": "data/weights/res_bn1_weight.bin",
                "bias_file": "data/weights/res_bn1_bias.bin",
                "mean_file": "data/weights/res_bn1_mean.bin",
                "var_file": "data/weights/res_bn1_var.bin",
                "reference_file": "data/reference/ref_res_bn1.bin"
            },
            {
                "layer_name": "res_relu1",
                "layer_type": "ReLU",
                "input_dim": [1, 64, 16, 16],
                "output_dim": [1, 64, 16, 16],
                "attributes": {},
                "reference_file": "data/reference/ref_res_relu1.bin"
            },
            {
                "layer_name": "res_conv2",
                "layer_type": "Conv2D",
                "input_dim": [1, 64, 16, 16],
                "output_dim": [1, 64, 16, 16],
                "attributes": {"kernel_size": 3, "stride": 1, "padding": 1},
                "weight_file": "data/weights/res_conv2_weight.bin",
                "bias_file": None,
                "reference_file": "data/reference/ref_res_conv2.bin"
            },
            {
                "layer_name": "res_bn2",
                "layer_type": "BatchNorm",
                "input_dim": [1, 64, 16, 16],
                "output_dim": [1, 64, 16, 16],
                "attributes": {"num_features": 64, "eps": 1e-5},
                "weight_file": "data/weights/res_bn2_weight.bin",
                "bias_file": "data/weights/res_bn2_bias.bin",
                "mean_file": "data/weights/res_bn2_mean.bin",
                "var_file": "data/weights/res_bn2_var.bin",
                "reference_file": "data/reference/ref_res_bn2.bin"
            },
            {
                "layer_name": "res_add",
                "layer_type": "ResidualAdd",
                "input_dim": [1, 64, 16, 16],
                "output_dim": [1, 64, 16, 16],
                "attributes": {"shortcut_source": "ds1_relu"},
                "reference_file": "data/reference/ref_res_add.bin"
            },
            {
                "layer_name": "res_relu2",
                "layer_type": "ReLU",
                "input_dim": [1, 64, 16, 16],
                "output_dim": [1, 64, 16, 16],
                "attributes": {},
                "reference_file": "data/reference/ref_res_out.bin"
            },
            {
                "layer_name": "ds2_conv",
                "layer_type": "Conv2D",
                "input_dim": [1, 64, 16, 16],
                "output_dim": [1, 128, 8, 8],
                "attributes": {"kernel_size": 3, "stride": 2, "padding": 1},
                "weight_file": "data/weights/ds2_conv_weight.bin",
                "bias_file": None,
                "reference_file": "data/reference/ref_ds2_conv.bin"
            },
            {
                "layer_name": "ds2_bn",
                "layer_type": "BatchNorm",
                "input_dim": [1, 128, 8, 8],
                "output_dim": [1, 128, 8, 8],
                "attributes": {"num_features": 128, "eps": 1e-5},
                "weight_file": "data/weights/ds2_bn_weight.bin",
                "bias_file": "data/weights/ds2_bn_bias.bin",
                "mean_file": "data/weights/ds2_bn_mean.bin",
                "var_file": "data/weights/ds2_bn_var.bin",
                "reference_file": "data/reference/ref_ds2_bn.bin"
            },
            {
                "layer_name": "ds2_relu",
                "layer_type": "ReLU",
                "input_dim": [1, 128, 8, 8],
                "output_dim": [1, 128, 8, 8],
                "attributes": {},
                "reference_file": "data/reference/ref_ds2_out.bin"
            },
            {
                "layer_name": "gap",
                "layer_type": "GlobalAveragePool",
                "input_dim": [1, 128, 8, 8],
                "output_dim": [1, 128, 1, 1],
                "attributes": {},
                "reference_file": "data/reference/ref_gap_out.bin"
            },
            {
                "layer_name": "classifier",
                "layer_type": "FullyConnected",
                "input_dim": [1, 128],
                "output_dim": [1, 10],
                "attributes": {"in_features": 128, "out_features": 10},
                "weight_file": "data/weights/classifier_weight.bin",
                "bias_file": "data/weights/classifier_bias.bin",
                "reference_file": "data/reference/ref_logits.bin"
            },
            {
                "layer_name": "softmax",
                "layer_type": "Softmax",
                "input_dim": [1, 10],
                "output_dim": [1, 10],
                "attributes": {},
                "reference_file": "data/reference/ref_probabilities.bin"
            }
        ]
    }

    config_path = os.path.join(config_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"\n[OK] Model configuration saved to: {config_path}")
    print("=" * 70)

if __name__ == "__main__":
    export_all()