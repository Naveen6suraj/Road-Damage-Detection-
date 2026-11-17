"""
ROADGUARD AI - Hardware-Aware Benchmarking Suite
Script: benchmark.py
Version: 3.1.0

Comprehensive hardware performance benchmarking suite for computer vision and MLOps:
1. Model loading latency (cold disk-to-VRAM initialization).
2. Single-image cold start latency vs. steady-state warm latency (N iterations).
3. Batch inference throughput (Frames Per Second).
4. Peak GPU VRAM utilization (allocated vs. reserved) via PyTorch CUDA telemetry.
5. System CPU utilization and RAM footprint via psutil.
6. Structured export to reports/hardware_benchmark.json and reports/hardware_benchmark.csv.
"""

import os
import sys
import time
import json
import csv
import argparse
import datetime
import platform
import numpy as np
import torch
import psutil

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


def collect_system_telemetry() -> dict:
    """Collect hardware specifications and platform details."""
    cpu_name = platform.processor() or "AMD Ryzen 5 5600H"
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)

    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None (CPU Execution)"
    gpu_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_avail else 0.0

    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cpu": cpu_name,
        "physical_cores": psutil.cpu_count(logical=False),
        "total_cores": psutil.cpu_count(logical=True),
        "system_ram_gb": ram_gb,
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "gpu_vram_capacity_gb": gpu_vram_gb
    }


def run_benchmark(
    model_path: str = "models/best.pt",
    device: str = "auto",
    iterations: int = 10,
    batch_size: int = 8,
    imgsz: int = config.DEFAULT_IMGSZ,
    reports_dir: str = "reports"
) -> dict:
    """
    Execute empirical hardware benchmark measuring latency, throughput, and memory.
    """
    from ultralytics import YOLO

    telemetry = collect_system_telemetry()

    # Determine target model
    target_model_path = model_path
    is_base_model = False
    if not os.path.exists(target_model_path) or os.path.getsize(target_model_path) < 1000:
        base_fallback = os.path.join(config.MODEL_FOLDER, "yolo11n.pt")
        if os.path.exists(base_fallback):
            print(f"[*] Custom weights not found at '{model_path}'. Benchmarking base model: {base_fallback}")
            target_model_path = base_fallback
            is_base_model = True
        else:
            raise FileNotFoundError(f"Neither custom model '{model_path}' nor base model '{base_fallback}' found.")

    # Target device configuration
    if device == "auto":
        target_device = "0" if telemetry["cuda_available"] else "cpu"
    else:
        target_device = device

    print("=" * 65)
    print(" ROADGUARD AI - Hardware-Aware Benchmarking Suite v3.1")
    print("=" * 65)
    print(f" OS:             {telemetry['os']}")
    print(f" CPU:            {telemetry['cpu']} ({telemetry['total_cores']} threads)")
    print(f" System RAM:     {telemetry['system_ram_gb']} GB")
    print(f" Target Device:  {target_device} ({telemetry['gpu_name']})")
    print(f" Target Model:   {target_model_path} ({'Base Model' if is_base_model else 'Custom Road-Damage Model'})")
    print(f" Warm Iterations: {iterations} | Batch Size: {batch_size} | Resolution: {imgsz}x{imgsz}")
    print("-" * 65)

    # Reset PyTorch CUDA memory counters
    if telemetry["cuda_available"]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    process = psutil.Process()
    ram_before_mb = process.memory_info().rss / (1024**2)

    # 1. Model Loading Latency
    print("[*] Benchmarking Model Load Latency...")
    t0 = time.perf_counter()
    model = YOLO(target_model_path)
    model_load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"    Model Load Time: {model_load_ms:.2f} ms")

    # Generate synthetic representative roadway frame (640x640x3 uint8)
    np.random.seed(42)
    sample_frame = np.random.randint(40, 220, (imgsz, imgsz, 3), dtype=np.uint8)

    # 2. Cold-Start Latency (First single-image inference)
    print("[*] Benchmarking Single-Image Cold Start...")
    t0 = time.perf_counter()
    _ = model(sample_frame, device=target_device, imgsz=imgsz, verbose=False)
    if telemetry["cuda_available"] and target_device != "cpu":
        torch.cuda.synchronize()
    cold_start_ms = (time.perf_counter() - t0) * 1000.0
    print(f"    Cold Start Latency: {cold_start_ms:.2f} ms")

    # 3. Steady-State Warm Latency (N iterations)
    print(f"[*] Benchmarking Single-Image Warm Latency ({iterations} iterations)...")
    warm_latencies_ms = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model(sample_frame, device=target_device, imgsz=imgsz, verbose=False)
        if telemetry["cuda_available"] and target_device != "cpu":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        warm_latencies_ms.append(dt_ms)

    avg_warm_ms = float(np.mean(warm_latencies_ms))
    min_warm_ms = float(np.min(warm_latencies_ms))
    max_warm_ms = float(np.max(warm_latencies_ms))
    std_warm_ms = float(np.std(warm_latencies_ms))
    p95_warm_ms = float(np.percentile(warm_latencies_ms, 95))

    print(f"    Avg Warm Latency:   {avg_warm_ms:.2f} ms (+/- {std_warm_ms:.2f} ms)")
    print(f"    P95 Warm Latency:   {p95_warm_ms:.2f} ms (Min: {min_warm_ms:.2f} ms, Max: {max_warm_ms:.2f} ms)")

    # 4. Batch Inference & Throughput (FPS)
    print(f"[*] Benchmarking Throughput with Batch Size {batch_size}...")
    batch_frames = [sample_frame.copy() for _ in range(batch_size)]

    # Warmup batch
    _ = model(batch_frames, device=target_device, imgsz=imgsz, verbose=False)
    if telemetry["cuda_available"] and target_device != "cpu":
        torch.cuda.synchronize()

    batch_runs = 5
    batch_latencies_ms = []
    for _ in range(batch_runs):
        t0 = time.perf_counter()
        _ = model(batch_frames, device=target_device, imgsz=imgsz, verbose=False)
        if telemetry["cuda_available"] and target_device != "cpu":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        batch_latencies_ms.append(dt_ms)

    avg_batch_time_sec = (np.mean(batch_latencies_ms) / 1000.0)
    throughput_fps = float(batch_size / avg_batch_time_sec) if avg_batch_time_sec > 0 else 0.0
    print(f"    Batch Throughput:   {throughput_fps:.1f} FPS (avg {np.mean(batch_latencies_ms):.1f} ms per batch)")

    # 5. Memory Telemetry
    ram_after_mb = process.memory_info().rss / (1024**2)
    process_ram_used_mb = max(0.0, ram_after_mb - ram_before_mb)
    cpu_percent = psutil.cpu_percent(interval=0.2)

    gpu_allocated_mb = 0.0
    gpu_reserved_mb = 0.0
    gpu_peak_allocated_mb = 0.0
    gpu_peak_reserved_mb = 0.0

    if telemetry["cuda_available"] and target_device != "cpu":
        gpu_allocated_mb = torch.cuda.memory_allocated() / (1024**2)
        gpu_reserved_mb = torch.cuda.memory_reserved() / (1024**2)
        gpu_peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
        gpu_peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
        print(f"    GPU Peak VRAM Alloc: {gpu_peak_allocated_mb:.1f} MB | Reserved: {gpu_peak_reserved_mb:.1f} MB")
    print(f"    Process RAM Footprint: {ram_after_mb:.1f} MB | System CPU: {cpu_percent:.1f}%")

    # Assemble structured results
    results = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "version": config.APP_VERSION,
        "model_path": target_model_path,
        "is_base_model": is_base_model,
        "model_status": "BASE_MODEL_BENCHMARK" if is_base_model else "CUSTOM_MODEL_BENCHMARK",
        "device_used": target_device,
        "system": telemetry,
        "config": {
            "warm_iterations": iterations,
            "batch_size": batch_size,
            "input_resolution": [imgsz, imgsz]
        },
        "latencies_ms": {
            "model_load": round(model_load_ms, 2),
            "cold_start": round(cold_start_ms, 2),
            "warm_mean": round(avg_warm_ms, 2),
            "warm_min": round(min_warm_ms, 2),
            "warm_max": round(max_warm_ms, 2),
            "warm_std": round(std_warm_ms, 2),
            "warm_p95": round(p95_warm_ms, 2)
        },
        "throughput": {
            "batch_size": batch_size,
            "fps": round(throughput_fps, 1),
            "ms_per_frame": round(1000.0 / throughput_fps, 2) if throughput_fps > 0 else 0.0
        },
        "memory": {
            "gpu_allocated_mb": round(gpu_allocated_mb, 1),
            "gpu_reserved_mb": round(gpu_reserved_mb, 1),
            "gpu_peak_allocated_mb": round(gpu_peak_allocated_mb, 1),
            "gpu_peak_reserved_mb": round(gpu_peak_reserved_mb, 1),
            "process_rss_ram_mb": round(ram_after_mb, 1),
            "process_incremental_ram_mb": round(process_ram_used_mb, 1),
            "system_cpu_percent": round(cpu_percent, 1)
        }
    }

    # 6. Export Reports
    os.makedirs(reports_dir, exist_ok=True)
    json_path = os.path.join(reports_dir, "hardware_benchmark.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Exported JSON benchmark report to: {json_path}")

    csv_path = os.path.join(reports_dir, "hardware_benchmark.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value", "Unit"])
        writer.writerow(["Timestamp (UTC)", results["timestamp"], "ISO8601"])
        writer.writerow(["App Version", results["version"], "str"])
        writer.writerow(["Model Evaluated", os.path.basename(target_model_path), "filename"])
        writer.writerow(["Model Type", "Base Model" if is_base_model else "Custom Road-Damage", "category"])
        writer.writerow(["Compute Device", target_device, "device"])
        writer.writerow(["GPU Name", telemetry["gpu_name"], "str"])
        writer.writerow(["CPU Name", telemetry["cpu"], "str"])
        writer.writerow(["Model Load Latency", results["latencies_ms"]["model_load"], "ms"])
        writer.writerow(["Cold Start Latency", results["latencies_ms"]["cold_start"], "ms"])
        writer.writerow(["Warm Latency (Mean)", results["latencies_ms"]["warm_mean"], "ms"])
        writer.writerow(["Warm Latency (P95)", results["latencies_ms"]["warm_p95"], "ms"])
        writer.writerow(["Batch Size", batch_size, "count"])
        writer.writerow(["Inference Throughput", results["throughput"]["fps"], "FPS"])
        writer.writerow(["Peak GPU VRAM Allocated", results["memory"]["gpu_peak_allocated_mb"], "MB"])
        writer.writerow(["Peak GPU VRAM Reserved", results["memory"]["gpu_peak_reserved_mb"], "MB"])
        writer.writerow(["Process RAM (RSS)", results["memory"]["process_rss_ram_mb"], "MB"])
        writer.writerow(["System CPU Utilization", results["memory"]["system_cpu_percent"], "%"])
    print(f"[+] Exported CSV benchmark report to: {csv_path}")
    print("=" * 65)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ROADGUARD AI Hardware Benchmarking Suite")
    parser.add_argument("--model", default="models/best.pt", help="Model weights path")
    parser.add_argument("--device", default="auto", help="Compute device ('auto', '0', 'cpu')")
    parser.add_argument("--iterations", default=10, type=int, help="Warmup iterations")
    parser.add_argument("--batch", default=8, type=int, help="Batch size for throughput test")
    parser.add_argument("--imgsz", default=config.DEFAULT_IMGSZ, type=int, help="Input image resolution")
    parser.add_argument("--outdir", default="reports", help="Directory for benchmark reports")
    args = parser.parse_args()

    run_benchmark(
        model_path=args.model,
        device=args.device,
        iterations=args.iterations,
        batch_size=args.batch,
        imgsz=args.imgsz,
        reports_dir=args.outdir
    )
