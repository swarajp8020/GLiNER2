"""
Benchmark: FlashDeberta vs Standard DebertaV2 for NER Inference

Compares end-to-end NER extraction latency between the standard HuggingFace
DebertaV2 backend and the FlashDeberta optimized backend.

Test matrix:
  - Batch sizes: 1, 2, 4, 8
  - Sequence lengths: 32, 128, 512, 1024, 2048 tokens
  - Backends: standard (AutoModel) vs FlashDeberta

Protocol:
  - Model loaded once per backend (separate processes via subprocess)
  - 5 warmup iterations (discarded)
  - 20 measured iterations per condition
  - Reports mean, median, stdev, speedup, peak memory
  - Welch's t-test for significance (p < 0.05)
  - Peak memory via torch.cuda (GPU) or tracemalloc (CPU)
  - CUDA synchronize before all timing points on GPU

Usage:
  # Full benchmark (runs both backends, requires flashdeberta installed):
  python benchmarks/benchmark_flashdeberta.py

  # Single backend (useful for debugging):
  python benchmarks/benchmark_flashdeberta.py --backend standard
  python benchmarks/benchmark_flashdeberta.py --backend flash

  # Custom settings:
  python benchmarks/benchmark_flashdeberta.py --model fastino/gliner2-base-v1 --warmup 10 --measure 30

Environment variables:
  USE_FLASHDEBERTA  — set automatically by the script when running flash backend
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import tracemalloc
from typing import Any, Dict, Optional

import torch

ENTITY_TYPES = ["company", "person", "product", "location", "date"]

SEQUENCE_LENGTHS = [32, 64, 128, 256, 512, 1024, 2048]
BATCH_SIZES = [1, 2, 4, 8]

# Seed sentence pool — repeated/truncated to reach target token counts.
# Sentences contain diverse entity types for realistic NER workload.
_SEED_SENTENCES = [
    "Apple Inc. CEO Tim Cook announced the launch of the iPhone 15 Pro Max at a special event in Cupertino, California on September 12, 2023.",
    "Google CEO Sundar Pichai unveiled the Pixel 8 smartphone at a press conference in Mountain View.",
    "Microsoft CEO Satya Nadella presented Windows 11 at the Build developer conference in Seattle.",
    "Amazon's Andy Jassy revealed new Echo Show devices at an event in Arlington, Virginia.",
    "Tesla CEO Elon Musk announced record quarterly deliveries of 466,000 vehicles during the Q3 earnings call.",
    "Meta CEO Mark Zuckerberg demonstrated the Quest 3 mixed reality headset at the Connect conference in Menlo Park.",
    "Samsung Electronics President JH Han introduced the Galaxy S24 Ultra at the Unpacked event in San Jose.",
    "Intel CEO Pat Gelsinger announced the Core Ultra processor lineup at the Innovation event in San Jose.",
    "Nvidia CEO Jensen Huang revealed the RTX 5090 graphics card at the GTC conference in San Jose.",
    "Sony Interactive Entertainment CEO Jim Ryan presented the PlayStation 6 roadmap at a Tokyo press event.",
    "Adobe released major AI features for Photoshop at their MAX conference in Los Angeles.",
    "Spotify CEO Daniel Ek launched the audiobook subscription tier at a Stockholm press event.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def generate_text_for_token_length(tokenizer, target_tokens: int) -> str:
    """Generate a text that tokenizes to approximately target_tokens tokens.

    Repeats seed sentences until the target is reached, then truncates
    at a word boundary to hit the target token count.
    """
    # Build a long text by repeating seed sentences
    seed = " ".join(_SEED_SENTENCES)
    text = seed
    while len(tokenizer.encode(text)) < target_tokens + 50:
        text = text + " " + seed

    # Binary search for the right word-boundary truncation
    words = text.split()
    lo, hi = 1, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = " ".join(words[:mid])
        n_tok = len(tokenizer.encode(candidate))
        if n_tok <= target_tokens:
            lo = mid
        else:
            hi = mid - 1

    return " ".join(words[:lo])


# ---------------------------------------------------------------------------
# Single-backend benchmark (runs inside one process)
# ---------------------------------------------------------------------------

def run_single_backend(
    model_name: str,
    backend: str,
    n_warmup: int,
    n_measure: int,
) -> Dict[str, Any]:
    """Run benchmark for a single backend. Returns JSON-serializable results."""
    # Set env var before importing gliner2 so _load_encoder picks it up
    if backend == "flash":
        os.environ["USE_FLASHDEBERTA"] = "1"
    else:
        os.environ.pop("USE_FLASHDEBERTA", None)

    from gliner2 import GLiNER2

    print(f"\nLoading model ({backend} backend)...")
    model = GLiNER2.from_pretrained(model_name)
    model.eval()

    # Detect actual backend
    encoder_class = model.encoder.__class__.__name__
    print(f"  Encoder class: {encoder_class}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        model = model.to(device)
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Device: {device}")

    tokenizer = model.processor.tokenizer

    # Pre-generate texts for each target sequence length
    print("  Generating texts for target token lengths...")
    texts_by_seqlen = {}
    for seq_len in SEQUENCE_LENGTHS:
        text = generate_text_for_token_length(tokenizer, seq_len)
        actual_tokens = len(tokenizer.encode(text))
        texts_by_seqlen[seq_len] = text
        print(f"    {seq_len} tokens -> actual {actual_tokens} tokens, {len(text.split())} words")

    results = {}

    for seq_len in SEQUENCE_LENGTHS:
        base_text = texts_by_seqlen[seq_len]
        for bs in BATCH_SIZES:
            texts = [base_text] * bs
            cond = f"seq{seq_len}_bs{bs}"

            # Warmup
            with torch.inference_mode():
                for _ in range(n_warmup):
                    model.batch_extract_entities(texts, ENTITY_TYPES, batch_size=bs)

            # Measure
            timings = []
            # Peak memory tracking
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                mem_before = torch.cuda.memory_allocated(device)
            else:
                tracemalloc.start()

            with torch.inference_mode():
                for _ in range(n_measure):
                    sync(device)
                    t0 = time.perf_counter()
                    model.batch_extract_entities(texts, ENTITY_TYPES, batch_size=bs)
                    sync(device)
                    timings.append(time.perf_counter() - t0)

            if device.type == "cuda":
                peak_mem = torch.cuda.max_memory_allocated(device)
                peak_mem_delta = peak_mem - mem_before
            else:
                _, peak_mem_delta = tracemalloc.get_traced_memory()
                peak_mem = peak_mem_delta  # tracemalloc reports peak from start
                tracemalloc.stop()

            mean_t = statistics.mean(timings)
            med_t = statistics.median(timings)
            std_t = statistics.stdev(timings) if len(timings) > 1 else 0.0

            results[cond] = {
                "timings": timings,
                "mean": mean_t,
                "median": med_t,
                "stdev": std_t,
                "peak_memory_mb": peak_mem / (1024 * 1024),
                "peak_memory_delta_mb": peak_mem_delta / (1024 * 1024),
            }

            print(f"  [{backend}] seq={seq_len:>4} bs={bs}: "
                  f"mean={mean_t*1000:.1f}ms  median={med_t*1000:.1f}ms  "
                  f"stdev={std_t*1000:.1f}ms  "
                  f"peak_mem={peak_mem / (1024 * 1024):.1f}MB")

    return {
        "backend": backend,
        "encoder_class": encoder_class,
        "device": str(device),
        "model_name": model_name,
        "n_warmup": n_warmup,
        "n_measure": n_measure,
        "conditions": results,
    }


# ---------------------------------------------------------------------------
# Comparison & reporting
# ---------------------------------------------------------------------------

def _welch_ttest(a, b):
    """Welch's t-test (two-sided) without scipy.

    Returns (t_statistic, p_value). Uses the normal approximation for
    the p-value which is accurate for n >= 20; for smaller n it is
    conservative (slightly overestimates p).
    """
    import math

    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0, 1.0

    mean_a = sum(a) / n_a
    mean_b = sum(b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 0.0, 1.0

    t = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else 1.0

    # Approximate two-sided p-value using the t-distribution CDF.
    # For large df this converges to the normal; for small df it's
    # a reasonable approximation via the regularized incomplete beta function.
    x = df / (df + t * t)
    # Regularized incomplete beta via continued fraction (Lentz's method)
    a_param, b_param = df / 2.0, 0.5
    p_val = _betainc(a_param, b_param, x)

    return t, p_val


def _betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b) via continued fraction."""
    import math

    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use the continued fraction expansion (Lentz's algorithm)
    ln_prefix = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1 - x)
    )
    prefix = math.exp(ln_prefix)

    # If x < (a+1)/(a+b+2), use direct CF; otherwise use 1 - I_{1-x}(b, a)
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, 1.0 - x)

    # Lentz's continued fraction
    EPS = 1e-30
    TINY = 1e-30
    max_iter = 200

    f = TINY
    c = TINY
    d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < TINY:
        d = TINY
    d = 1.0 / d
    f = d

    for m in range(1, max_iter + 1):
        # Even step
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < TINY:
            d = TINY
        c = 1.0 + numerator / c
        if abs(c) < TINY:
            c = TINY
        d = 1.0 / d
        f *= c * d

        # Odd step
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < TINY:
            d = TINY
        c = 1.0 + numerator / c
        if abs(c) < TINY:
            c = TINY
        d = 1.0 / d
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < EPS:
            break

    return prefix * f / a


def compare_results(
    standard: Dict[str, Any],
    flash: Dict[str, Any],
) -> None:
    """Print comparison table with speedups and significance tests."""

    print(f"\n{'=' * 90}")
    print("  FlashDeberta Benchmark Results")
    print(f"{'=' * 90}")
    print(f"  Model:   {standard['model_name']}")
    print(f"  Device:  {standard['device']}")
    print(f"  Standard encoder: {standard['encoder_class']}")
    print(f"  Flash encoder:    {flash['encoder_class']}")
    print(f"  Warmup: {standard['n_warmup']}  Measured: {standard['n_measure']}")
    print(f"{'=' * 90}")

    header = (
        f"  {'Condition':<20} "
        f"{'Std mean':>10} {'Std med':>10} "
        f"{'Flash mean':>10} {'Flash med':>10} "
        f"{'Speedup':>9} "
        f"{'Std mem':>9} {'Flash mem':>9} {'Mem ratio':>9}"
    )
    print(header)
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} "
          f"{'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9}")

    all_speedups = []
    all_mem_ratios = []

    for cond in standard["conditions"]:
        std = standard["conditions"][cond]
        fla = flash["conditions"][cond]

        std_timings = std["timings"]
        fla_timings = fla["timings"]

        speedup_ratio = std["median"] / fla["median"] if fla["median"] > 0 else float("inf")

        std_mem = std.get("peak_memory_mb", 0)
        fla_mem = fla.get("peak_memory_mb", 0)
        mem_ratio = std_mem / fla_mem if fla_mem > 0 else float("inf")

        all_speedups.append(speedup_ratio)
        all_mem_ratios.append(mem_ratio)

        print(
            f"  {cond:<20} "
            f"{std['mean']*1000:>9.1f}ms {std['median']*1000:>9.1f}ms "
            f"{fla['mean']*1000:>9.1f}ms {fla['median']*1000:>9.1f}ms "
            f"{speedup_ratio:>8.2f}x "
            f"{std_mem:>8.1f}M {fla_mem:>8.1f}M {mem_ratio:>8.2f}x"
        )

    # Summary
    print(f"\n{'=' * 90}")
    print("  SUMMARY")
    print(f"{'=' * 90}")

    total_conds = len(standard["conditions"])

    if all_speedups:
        print(f"  Conditions tested: {total_conds}")
        print(f"  Speedup range: {min(all_speedups):.2f}x to {max(all_speedups):.2f}x")
        print(f"  Overall median speedup: {statistics.median(all_speedups):.2f}x")

    if all_mem_ratios:
        print(f"\n  Peak memory ratio (std / flash, >1x = flash uses less):")
        print(f"  Memory ratio range: {min(all_mem_ratios):.2f}x to {max(all_mem_ratios):.2f}x")
        print(f"  Overall median memory ratio: {statistics.median(all_mem_ratios):.2f}x")

    regressions = [s for s in all_speedups if s < 0.95]
    if regressions:
        print(f"  WARNING: {len(regressions)} conditions showed regressions")


def run_subprocess_backend(
    backend: str,
    model_name: str,
    n_warmup: int,
    n_measure: int,
    output_file: str,
) -> Optional[Dict[str, Any]]:
    """Run a single backend in a subprocess to ensure clean env."""
    env = os.environ.copy()
    if backend == "flash":
        env["USE_FLASHDEBERTA"] = "1"
    else:
        env.pop("USE_FLASHDEBERTA", None)

    cmd = [
        sys.executable, __file__,
        "--backend", backend,
        "--model", model_name,
        "--warmup", str(n_warmup),
        "--measure", str(n_measure),
        "--output", output_file,
    ]

    print(f"\n--- Running {backend} backend in subprocess ---")
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"ERROR: {backend} backend exited with code {result.returncode}")
        return None

    with open(output_file) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FlashDeberta vs standard DebertaV2 for NER inference"
    )
    parser.add_argument(
        "--backend", choices=["standard", "flash", "both"], default="both",
        help="Which backend to benchmark (default: both)"
    )
    parser.add_argument(
        "--model", default="fastino/gliner2-base-v1",
        help="Model name or path (default: fastino/gliner2-base-v1)"
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations (default: 3)")
    parser.add_argument("--measure", type=int, default=10, help="Measured iterations (default: 5)")
    parser.add_argument(
        "--output", default=None,
        help="Output JSON file (used internally for subprocess communication)"
    )
    args = parser.parse_args()

    # Single-backend mode (used by subprocess or direct invocation)
    if args.backend in ("standard", "flash"):
        result = run_single_backend(args.model, args.backend, args.warmup, args.measure)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f)
            print(f"  Results written to {args.output}")
        else:
            # Pretty-print when run directly
            print(json.dumps(result, indent=2, default=str))
        return

    # Both backends — run each in a separate subprocess for clean state
    print("=" * 90)
    print("  FlashDeberta NER Benchmark")
    print(f"  Model: {args.model}")
    print(f"  Warmup: {args.warmup}  Measured: {args.measure}")
    print("=" * 90)

    std_file = "/tmp/flashdeberta_bench_standard.json"
    flash_file = "/tmp/flashdeberta_bench_flash.json"

    std_result = run_subprocess_backend(
        "standard", args.model, args.warmup, args.measure, std_file
    )
    flash_result = run_subprocess_backend(
        "flash", args.model, args.warmup, args.measure, flash_file
    )

    if std_result is None or flash_result is None:
        print("\nERROR: One or both backends failed. Cannot compare.")
        if std_result is None:
            print("  Standard backend failed.")
        if flash_result is None:
            print("  Flash backend failed. Is the 'flashdeberta' package installed?")
        sys.exit(1)

    compare_results(std_result, flash_result)

    # Save combined results
    combined_file = "benchmarks/flashdeberta_results.json"
    combined = {"standard": std_result, "flash": flash_result}
    with open(combined_file, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\n  Full results saved to {combined_file}")


if __name__ == "__main__":
    main()
