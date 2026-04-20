"""
Latency benchmark for the inference endpoint.

Measures inference time on a batch of random images to establish
the latency baseline before and after any optimizations (ONNX export, etc).

Usage:
    python -m evaluation.benchmark --n 100
"""

import argparse
import time
import numpy as np
from inference.predict import DefectPredictor


def benchmark(n_images: int = 100, image_size: int = 256):
    predictor = DefectPredictor()

    # Disable heatmap during benchmarking — measures pure inference speed
    predictor.include_heatmap = False

    latencies = []
    print(f"Benchmarking {n_images} images on {predictor.device}...")

    for i in range(n_images):
        # Fake a random test image — we're measuring latency, not accuracy
        img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        result = predictor.predict(img)
        latencies.append(result["latency_ms"])

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_images} complete...")

    latencies = np.array(latencies)
    print(f"\n--- Benchmark Results ---")
    print(f"Mean latency:   {latencies.mean():.2f} ms")
    print(f"Median latency: {np.median(latencies):.2f} ms")
    print(f"P95 latency:    {np.percentile(latencies, 95):.2f} ms")
    print(f"P99 latency:    {np.percentile(latencies, 99):.2f} ms")
    print(f"Min / Max:      {latencies.min():.2f} / {latencies.max():.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of images to benchmark")
    args = parser.parse_args()
    benchmark(n_images=args.n)
