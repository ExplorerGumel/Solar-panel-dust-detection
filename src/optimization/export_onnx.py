import os
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from ..models.swin_classifier import build_model

def export_to_onnx(model, save_path="./model_store/swin_solar_panel.onnx", img_size=224):
    """
    Exports PyTorch model to ONNX format with dynamic batch axis.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False
    )

    # Verify ONNX model integrity
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"[SUCCESS] ONNX Model exported to: {save_path} ({file_size_mb:.2f} MB)")
    return save_path

def quantize_onnx_int8(onnx_path, save_path="./model_store/swin_solar_panel_int8.onnx"):
    """
    Applies Dynamic INT8 Quantization to ONNX model.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    quantize_dynamic(
        model_input=onnx_path,
        model_output=save_path,
        weight_type=QuantType.QUInt8
    )

    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"[SUCCESS] INT8 Quantized ONNX Model saved to: {save_path} ({file_size_mb:.2f} MB)")
    return save_path

def benchmark_onnx_models(fp32_path, int8_path, num_runs=100, img_size=224):
    """
    Runs benchmark comparing FP32 vs INT8 ONNX Runtime Latency and Memory footprint.
    """
    dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    sess_fp32 = ort.InferenceSession(fp32_path, providers=['CPUExecutionProvider'])
    sess_int8 = ort.InferenceSession(int8_path, providers=['CPUExecutionProvider'])

    # Warmup
    for _ in range(10):
        sess_fp32.run(None, {'input': dummy_input})
        sess_int8.run(None, {'input': dummy_input})

    # FP32 Benchmark
    t0 = time.time()
    for _ in range(num_runs):
        out_fp32 = sess_fp32.run(None, {'input': dummy_input})[0]
    fp32_latency = ((time.time() - t0) / num_runs) * 1000.0  # ms

    # INT8 Benchmark
    t0 = time.time()
    for _ in range(num_runs):
        out_int8 = sess_int8.run(None, {'input': dummy_input})[0]
    int8_latency = ((time.time() - t0) / num_runs) * 1000.0  # ms

    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
    int8_size = os.path.getsize(int8_path) / (1024 * 1024)

    speedup = fp32_latency / int8_latency
    size_reduction = ((fp32_size - int8_size) / fp32_size) * 100.0

    print("\n=======================================================")
    print("EDGE AI PERFORMANCE BENCHMARK (FP32 vs INT8 Quantization)")
    print("=======================================================")
    print(f"FP32 Model Size    : {fp32_size:.2f} MB")
    print(f"INT8 Model Size    : {int8_size:.2f} MB ({size_reduction:.1f}% reduction)")
    print(f"FP32 Latency       : {fp32_latency:.2f} ms/image ({1000.0/fp32_latency:.1f} FPS)")
    print(f"INT8 Latency       : {int8_latency:.2f} ms/image ({1000.0/int8_latency:.1f} FPS)")
    print(f"Latency Speedup    : {speedup:.2f}x faster on CPU!")
    print("=======================================================\n")

    return {
        'fp32_size_mb': fp32_size,
        'int8_size_mb': int8_size,
        'fp32_latency_ms': fp32_latency,
        'int8_latency_ms': int8_latency,
        'speedup': speedup
    }

if __name__ == "__main__":
    # Self-test export with resnet18/swin_t
    model, _ = build_model('resnet18', num_classes=6, pretrained=False)
    fp32_p = export_to_onnx(model, "./model_store/test_fp32.onnx")
    int8_p = quantize_onnx_int8(fp32_p, "./model_store/test_int8.onnx")
    benchmark_onnx_models(fp32_p, int8_p, num_runs=20)
