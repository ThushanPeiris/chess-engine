"""
Quantizes value_network.onnx to int8 and benchmarks it against the
original FP32 version. Given the model is only ~800KB - trivially small
against the 50MB submission cap - the only reason to quantize at all is
potential inference SPEED, not file size. This script measures that
directly rather than assuming quantization helps: for a network this
small, the overhead of int8 (de)quantization can sometimes offset or
even exceed the compute savings, unlike with much larger models where
the benefit is usually clear-cut.
"""

import time
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

INPUT_MODEL = "value_network.onnx"
OUTPUT_MODEL = "value_network_int8.onnx"
NUM_BENCHMARK_RUNS = 5000


def quantize():
    quantize_dynamic(
        model_input=INPUT_MODEL,
        model_output=OUTPUT_MODEL,
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized model saved to {OUTPUT_MODEL}")


def benchmark(model_path, label):
    session = ort.InferenceSession(model_path)
    dummy_input = np.random.rand(1, 771).astype(np.float32)

    # A few warm-up calls before timing - the first call or two often
    # includes one-time setup cost that isn't representative of steady
    # state repeated calls (which is what search actually does, calling
    # evaluate() many thousands of times per move).
    for _ in range(20):
        session.run(None, {"board_features": dummy_input})

    start = time.time()
    for _ in range(NUM_BENCHMARK_RUNS):
        session.run(None, {"board_features": dummy_input})
    elapsed = time.time() - start

    per_call_us = (elapsed / NUM_BENCHMARK_RUNS) * 1_000_000
    print(f"{label}: {elapsed:.3f}s for {NUM_BENCHMARK_RUNS} calls "
          f"({per_call_us:.1f} microseconds/call)")
    return per_call_us


if __name__ == "__main__":
    print(f"Original model size: {__import__('os').path.getsize(INPUT_MODEL) / 1024:.1f} KB")

    quantize()

    print(f"Quantized model size: {__import__('os').path.getsize(OUTPUT_MODEL) / 1024:.1f} KB")
    print()

    fp32_time = benchmark(INPUT_MODEL, "FP32 (original)")
    int8_time = benchmark(OUTPUT_MODEL, "INT8 (quantized)")

    print()
    if int8_time < fp32_time:
        speedup = fp32_time / int8_time
        print(f"INT8 is {speedup:.2f}x faster - worth using {OUTPUT_MODEL} in the agent.")
    else:
        slowdown = int8_time / fp32_time
        print(f"INT8 is actually {slowdown:.2f}x SLOWER for this small a network - "
              f"stick with the original {INPUT_MODEL} instead. This can happen "
              f"when a model is small enough that quantization overhead outweighs "
              f"the compute savings.")