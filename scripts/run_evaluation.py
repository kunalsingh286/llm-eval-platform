import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from evals.faithfulness import FaithfulnessEvaluator
from evals.relevance import RelevanceEvaluator
from evals.format_accuracy import FormatAccuracyEvaluator
from tracking.mlflow_logger import MLflowLogger


def load_outputs():
    path = ROOT / "runs" / "output" / "baseline_outputs_v1.json"
    return json.loads(path.read_text())


def load_dataset():
    path = ROOT / "data" / "golden" / "dataset_v1.json"
    return json.loads(path.read_text())["samples"]


def aggregate_scores(results):
    totals = {}
    counts = {}

    for r in results:
        for k, v in r["scores"].items():
            totals[k] = totals.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1

    return {k: round(totals[k] / counts[k], 3) for k in totals}


def main():
    outputs = load_outputs()
    dataset = {s["id"]: s for s in load_dataset()}

    evaluators = [
        FaithfulnessEvaluator(),
        RelevanceEvaluator(),
        FormatAccuracyEvaluator(),
    ]

    results = []

    for item in outputs:
        sample = dataset[item["id"]]
        sample_scores = {}

        for evaluator in evaluators:
            score = evaluator.score(
                input_text=sample["input"],
                model_output=item["output"],
                expected_output=sample["expected_output"],
            )
            sample_scores.update(score)

        results.append(
            {
                "id": item["id"],
                "scores": sample_scores,
            }
        )

    out_path = ROOT / "runs" / "results" / "eval_results_v1.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    aggregate = aggregate_scores(results)

    # ---------------- MLflow Tracking ----------------
    logger = MLflowLogger(experiment_name="llm-eval-platform")
    logger.start_run(run_name="baseline_eval_v1")

    logger.log_params(
        {
            "prompt_version": "v1",
            "dataset_version": "v1",
            "model": "llama3.1:8b",
        }
    )

    logger.log_metrics(aggregate)
    logger.log_artifact(str(out_path))

    logger.end_run()

    print("Aggregate metrics:", aggregate)
    print(f"Saved evaluation results to {out_path}")


if __name__ == "__main__":
    main()
