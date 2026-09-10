# src/dump_weights.py
import os
import json
import struct
import torch
from src.model import MiniResNet

def dump_bare_metal_weights(
    weights_path="./artifacts/optimizer_sgd_momentum_best.pt",
    output_dir="./artifacts/bare_metal"
):
    os.makedirs(output_dir, exist_ok=True)
    bin_path = os.path.join(output_dir, "model_weights.bin")
    manifest_path = os.path.join(output_dir, "weights_manifest.json")

    # Load the trained champion weights
    model = MiniResNet(num_classes=10)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    manifest = []
    current_byte_offset = 0
    total_floats = 0

    with open(bin_path, "wb") as bin_file:
        for name, param in model.state_dict().items():
            tensor = param.detach().cpu().float().numpy()
            flattened = tensor.flatten()
            num_elements = len(flattened)
            byte_size = num_elements * 4  # 4 bytes per float32

            # Pack raw floats in little-endian IEEE-754 format
            raw_bytes = struct.pack(f"<{num_elements}f", *flattened)
            bin_file.write(raw_bytes)

            manifest.append({
                "layer_name": name,
                "shape": list(tensor.shape),
                "num_elements": num_elements,
                "byte_offset": current_byte_offset,
                "byte_size": byte_size
            })

            current_byte_offset += byte_size
            total_floats += num_elements

    # Save the manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_kb = current_byte_offset / 1024
    print("==========================================================")
    print("Bare-Metal Serialization Complete:")
    print(f" -> Binary file:   {bin_path} ({total_kb:.2f} KB)")
    print(f" -> Manifest file: {manifest_path}")
    print(f" -> Total Floats:  {total_floats:,} parameters")
    print("==========================================================")

if __name__ == "__main__":
    dump_bare_metal_weights()