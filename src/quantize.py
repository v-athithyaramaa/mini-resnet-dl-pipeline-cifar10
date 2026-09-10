# src/quantize.py
import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def run_quantization(
    input_model="./artifacts/cifar10_miniresnet.onnx",
    output_model="./artifacts/cifar10_miniresnet_int8.onnx"
):
    print("Running Dynamic INT8 Quantization...")
    quantize_dynamic(
        model_input=input_model,
        model_output=output_model,
        weight_type=QuantType.QUInt8
    )

    fp32_kb = os.path.getsize(input_model) / 1024
    int8_kb = os.path.getsize(output_model) / 1024
    compression_ratio = fp32_kb / int8_kb

    print(f"FP32 Model Size: {fp32_kb:.2f} KB")
    print(f"INT8 Model Size: {int8_kb:.2f} KB")
    print(f"Storage Reduction: {compression_ratio:.2f}x smaller")

if __name__ == "__main__":
    run_quantization()