"""Benchmark GLiNER2 quantization and compilation on academic NER datasets.

Evaluates up to 6 conditions (CPU fp32/fp16, GPU fp32/fp16/compile/fp16+compile)
on CoNLL-2003 and WNUT-2017 test sets using nervaluate strict entity matching.

Uses the evaluation infrastructure from knowledge_engine (sibling directory).

Usage:
    python benchmark_quantization.py [--sample-size 200] [--threshold 0.5]
"""

import argparse
import gc
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

# Add knowledge_engine to path for evaluation infrastructure
KE_ROOT = Path(__file__).resolve().parent.parent / "knowledge_engine"
sys.path.insert(0, str(KE_ROOT))

from src.core.entities import Document, Entity
from src.core.types import EntityType
from src.datasets.loaders import load_conll2003, load_wnut2017
from src.evaluation.metrics import evaluate_entities

from gliner2 import GLiNER2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_NAME = "fastino/gliner2-base-v1"

# GLiNER2 uses lowercase labels; map to EntityType for nervaluate
LABEL_TO_ENTITY_TYPE = {
    "person": EntityType.PERSON,
    "organization": EntityType.ORGANIZATION,
    "location": EntityType.LOCATION,
    "product": EntityType.PRODUCT,
}

# Entity types evaluated per dataset
DATASET_ENTITY_TYPES = {
    "CoNLL-2003": {
        "labels": ["person", "organization", "location"],
        "eval_types": ["PERSON", "ORGANIZATION", "LOCATION"],
        "loader": load_conll2003,
    },
    "WNUT-2017": {
        "labels": ["person", "organization", "location", "product"],
        "eval_types": ["PERSON", "ORGANIZATION", "LOCATION", "PRODUCT"],
        "loader": load_wnut2017,
    },
}


@dataclass
class ConditionResult:
    """Result for one (condition, dataset) pair."""

    condition: str
    dataset: str
    device: str
    quantization: str
    num_documents: int
    num_gold_entities: int
    num_pred_entities: int
    precision: float
    recall: float
    f1: float
    total_seconds: float
    per_doc_seconds: float
    docs_per_second: float
    std_seconds: float
    by_entity_type: dict


def _gliner2_predict(
    model: GLiNER2,
    gold_data: list[tuple[Document, list[Entity]]],
    labels: list[str],
    threshold: float,
    warmup: int = 3,
) -> tuple[list[tuple[Document, list[Entity]]], dict]:
    """Run GLiNER2 inference and return predictions + timing stats.

    Interleaves warmup within the same process to avoid cache-warming bias.
    """
    docs = [doc for doc, _ in gold_data]

    # Warmup (not timed)
    if docs:
        for _ in range(warmup):
            model.extract_entities(docs[0].text, labels, threshold=threshold,
                                   include_spans=True, include_confidence=True)

    # Timed inference — per-document
    predictions = []
    doc_times = []

    for doc, _ in gold_data:
        start = time.perf_counter()
        result = model.extract_entities(
            doc.text, labels, threshold=threshold,
            include_spans=True, include_confidence=True,
        )
        elapsed = time.perf_counter() - start
        doc_times.append(elapsed)

        # Convert GLiNER2 output → list[Entity]
        entities = []
        ent_dict = result.get("entities", {})
        for label_name, ent_list in ent_dict.items():
            etype = LABEL_TO_ENTITY_TYPE.get(label_name)
            if etype is None:
                continue
            for ent in ent_list:
                try:
                    entities.append(Entity(
                        text=ent["text"],
                        label=etype,
                        start=ent["start"],
                        end=ent["end"],
                        confidence=ent.get("confidence", 1.0),
                        source="gliner2",
                    ))
                except (ValueError, KeyError):
                    continue

        predictions.append((doc, entities))

    total_time = sum(doc_times)
    n = len(doc_times)
    timing = {
        "total_seconds": total_time,
        "per_doc_seconds": total_time / n if n else 0.0,
        "docs_per_second": n / total_time if total_time > 0 else 0.0,
        "std_seconds": statistics.stdev(doc_times) if n > 1 else 0.0,
    }
    return predictions, timing


def evaluate_condition(
    condition_name: str,
    model: GLiNER2,
    dataset_name: str,
    gold_data: list[tuple[Document, list[Entity]]],
    labels: list[str],
    eval_types: list[str],
    threshold: float,
    device_str: str,
    quant_str: str,
) -> ConditionResult:
    """Evaluate one condition on one dataset."""
    logger.info(f"  {condition_name} / {dataset_name}: running inference on {len(gold_data)} docs ...")
    predictions, timing = _gliner2_predict(model, gold_data, labels, threshold)

    logger.info(f"  {condition_name} / {dataset_name}: evaluating ...")
    results = evaluate_entities(gold_data, predictions, entity_types=eval_types)

    num_gold = sum(len(ents) for _, ents in gold_data)
    num_pred = sum(len(ents) for _, ents in predictions)

    by_type = {}
    for etype, type_res in results.get("by_entity_type", {}).items():
        s = type_res["strict"]
        by_type[etype] = {
            "precision": s["precision"],
            "recall": s["recall"],
            "f1": s["f1"],
        }

    return ConditionResult(
        condition=condition_name,
        dataset=dataset_name,
        device=device_str,
        quantization=quant_str,
        num_documents=len(gold_data),
        num_gold_entities=num_gold,
        num_pred_entities=num_pred,
        precision=results["strict"]["precision"],
        recall=results["strict"]["recall"],
        f1=results["strict"]["f1"],
        total_seconds=timing["total_seconds"],
        per_doc_seconds=timing["per_doc_seconds"],
        docs_per_second=timing["docs_per_second"],
        std_seconds=timing["std_seconds"],
        by_entity_type=by_type,
    )


def _load_model(device: str, quantize: bool, compile: bool = False) -> GLiNER2:
    """Load model with specified configuration.

    Args:
        device: "cpu" or "cuda"
        quantize: Whether to apply fp16 via model.quantize()
        compile: Whether to torch.compile encoder and span-rep
    """
    model = GLiNER2.from_pretrained(
        MODEL_NAME,
        map_location=device if device != "cpu" else None,
        quantize=quantize,
        compile=compile,
    )
    if compile:
        # Warmup the compiled graphs with a dummy call so tracing
        # overhead doesn't pollute the timed runs.
        model.extract_entities("warmup text", ["person"], threshold=0.5)
    return model


def _free_model(model):
    """Aggressively free model memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def build_conditions(has_cuda: bool) -> list[dict]:
    """Build list of conditions to benchmark."""
    conditions = [
        {"name": "CPU-fp32", "device": "cpu", "quantize": False, "quant_label": "none"},
        {"name": "CPU-fp16", "device": "cpu", "quantize": True, "quant_label": "fp16"},
    ]
    if has_cuda:
        conditions.extend([
            {"name": "GPU-fp32", "device": "cuda", "quantize": False, "quant_label": "none"},
            {"name": "GPU-fp16", "device": "cuda", "quantize": True, "quant_label": "fp16"},
            {"name": "GPU-compile", "device": "cuda", "quantize": False, "compile": True, "quant_label": "compile"},
            {"name": "GPU-fp16+comp", "device": "cuda", "quantize": True, "compile": True, "quant_label": "fp16+compile"},
        ])
    return conditions


def print_summary_table(all_results: list[ConditionResult]):
    """Print a summary comparison table."""
    # Group by dataset
    datasets = sorted(set(r.dataset for r in all_results))

    for ds in datasets:
        ds_results = [r for r in all_results if r.dataset == ds]
        print(f"\n{'=' * 90}")
        print(f"  {ds}")
        print(f"{'=' * 90}")
        print(f"  {'Condition':<16} {'F1':>7} {'Prec':>7} {'Rec':>7} {'docs/s':>8} {'ms/doc':>8} {'std':>7}  {'Gold':>5} {'Pred':>5}")
        print(f"  {'-'*16} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*7}  {'-'*5} {'-'*5}")
        for r in ds_results:
            ms_per_doc = r.per_doc_seconds * 1000
            std_ms = r.std_seconds * 1000
            print(
                f"  {r.condition:<16} {r.f1:>7.4f} {r.precision:>7.4f} {r.recall:>7.4f} "
                f"{r.docs_per_second:>8.1f} {ms_per_doc:>8.1f} {std_ms:>7.1f}  "
                f"{r.num_gold_entities:>5} {r.num_pred_entities:>5}"
            )

    # Speedup summary
    print(f"\n{'=' * 90}")
    print(f"  Speedup vs baseline (docs/sec ratio)")
    print(f"{'=' * 90}")
    for ds in datasets:
        ds_results = [r for r in all_results if r.dataset == ds]
        cpu_baseline = next((r for r in ds_results if r.condition == "CPU-fp32"), None)
        gpu_baseline = next((r for r in ds_results if r.condition == "GPU-fp32"), None)

        print(f"\n  {ds}:")
        for r in ds_results:
            if r.condition == "CPU-fp32":
                continue
            if r.device == "cpu" and cpu_baseline:
                ratio = r.docs_per_second / cpu_baseline.docs_per_second if cpu_baseline.docs_per_second > 0 else 0
                print(f"    {r.condition:<16} {ratio:.2f}x vs CPU-fp32")
            elif r.device == "cuda" and gpu_baseline:
                ratio = r.docs_per_second / gpu_baseline.docs_per_second if gpu_baseline.docs_per_second > 0 else 0
                print(f"    {r.condition:<16} {ratio:.2f}x vs GPU-fp32")


def main():
    parser = argparse.ArgumentParser(description="Benchmark GLiNER2 quantization accuracy and throughput")
    parser.add_argument("--sample-size", type=int, default=200, help="Documents per dataset (default: 200)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Entity confidence threshold (default: 0.5)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: auto-generated)")
    args = parser.parse_args()

    has_cuda = torch.cuda.is_available()

    logger.info(f"CUDA: {has_cuda}, sample_size: {args.sample_size}")
    if has_cuda:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load datasets once
    logger.info("Loading datasets ...")
    datasets = {}
    for ds_name, ds_cfg in DATASET_ENTITY_TYPES.items():
        data = ds_cfg["loader"]("test", sample_size=args.sample_size)
        datasets[ds_name] = data
        n_ents = sum(len(ents) for _, ents in data)
        logger.info(f"  {ds_name}: {len(data)} docs, {n_ents} gold entities")

    # Build conditions
    conditions = build_conditions(has_cuda)
    logger.info(f"Conditions: {[c['name'] for c in conditions]}")

    # Run benchmarks
    all_results: list[ConditionResult] = []

    for cond in conditions:
        cond_name = cond["name"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading model: {cond_name} ...")

        model = _load_model(
            device=cond["device"],
            quantize=cond["quantize"],
            compile=cond.get("compile", False),
        )

        for ds_name, ds_cfg in DATASET_ENTITY_TYPES.items():
            gold_data = datasets[ds_name]
            result = evaluate_condition(
                condition_name=cond_name,
                model=model,
                dataset_name=ds_name,
                gold_data=gold_data,
                labels=ds_cfg["labels"],
                eval_types=ds_cfg["eval_types"],
                threshold=args.threshold,
                device_str=cond["device"],
                quant_str=cond.get("quant_label", "none"),
            )
            all_results.append(result)
            logger.info(
                f"  -> F1={result.f1:.4f}  P={result.precision:.4f}  R={result.recall:.4f}  "
                f"{result.docs_per_second:.1f} docs/s"
            )

        _free_model(model)

    # Print summary
    print_summary_table(all_results)

    # Save JSON
    output_path = args.output or f"quantization_benchmark_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "sample_size": args.sample_size,
        "threshold": args.threshold,
        "cuda_available": has_cuda,
        "cuda_device": torch.cuda.get_device_name(0) if has_cuda else None,
        "pytorch_version": torch.__version__,
        "results": [asdict(r) for r in all_results],
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
