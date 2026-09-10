# src/benchmark.py
import os
import time
import numpy as np
import torch
import onnxruntime as ort
from src.model import MiniResNet

def benchmark_pytorch(weights_path, runs=200, warmup=20):
    model = MiniResNet(num_classes=10)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    
    dummy = torch.randn(1, 3, 32, 32)
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy)
            
    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        timings.append((time.perf_counter() - t0) * 1000)
        
    return np.mean(timings), np.percentile(timings, 50), np.percentile(timings, 99)

def benchmark_onnx(onnx_path, runs=200, warmup=20):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, 32, 32).astype(np.float32)
    
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy})
        
    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy})
        timings.append((time.perf_counter() - t0) * 1000)
        
    return np.mean(timings), np.percentile(timings, 50), np.percentile(timings, 99)

if __name__ == "__main__":
    pt_weights = "./artifacts/optimizer_sgd_momentum_best.pt"
    onnx_fp32 = "./artifacts/cifar10_miniresnet.onnx"
    onnx_int8 = "./artifacts/cifar10_miniresnet_int8.onnx"
    
    print("==================================================================")
    print("1. Benchmarking PyTorch Eager (FP32)...")
    pt_mean, pt_p50, pt_p99 = benchmark_pytorch(pt_weights)
    print(f"   Mean: {pt_mean:.3f} ms | P50: {pt_p50:.3f} ms | P99: {pt_p99:.3f} ms")

    print("\n2. Benchmarking ONNX Runtime (FP32 Engine)...")
    ort_mean, ort_p50, ort_p99 = benchmark_onnx(onnx_fp32)
    print(f"   Mean: {ort_mean:.3f} ms | P50: {ort_p50:.3f} ms | P99: {ort_p99:.3f} ms")

    if os.path.exists(onnx_int8):
        print("\n3. Benchmarking ONNX Runtime (INT8 Dynamic Quantized)...")
        int8_mean, int8_p50, int8_p99 = benchmark_onnx(onnx_int8)
        print(f"   Mean: {int8_mean:.3f} ms | P50: {int8_p50:.3f} ms | P99: {int8_p99:.3f} ms")
    print("==================================================================")