import json
import time

import numpy as np
from js import Uint8Array, reduce_colorspace
from PIL import Image


def generate_test_image(width, height):
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


def benchmark_sample_rate(width, height, num_colors, sample_rate):
    img = generate_test_image(width, height)
    flat_img = img.flatten().tolist()

    start_time = time.time()
    reduce_colorspace(width, height, Uint8Array.new(flat_img), num_colors, sample_rate)
    end_time = time.time()

    return end_time - start_time


def run_benchmarks():
    image_sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
    color_counts = [8, 16, 32, 64]
    sample_rates = [1, 2, 4, 8, 16]

    results = []

    for width, height in image_sizes:
        for num_colors in color_counts:
            for sample_rate in sample_rates:
                time_taken = benchmark_sample_rate(
                    width, height, num_colors, sample_rate
                )
                results.append(
                    {
                        "width": width,
                        "height": height,
                        "num_colors": num_colors,
                        "sample_rate": sample_rate,
                        "time": time_taken,
                    }
                )

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f)


run_benchmarks()
