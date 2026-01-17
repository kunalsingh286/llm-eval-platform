import sys
import json
from pathlib import Path

# -------------------------------------------------
# Project root resolution
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from evals.faithfulness import FaithfulnessEvaluator
from evals.relevance import RelevanceEvaluator
from evals.format_accuracy import FormatAccuracyEvaluator
from evals.regression import RegressionDetector
from tracking.mlflow_logger import MLflowLogger


# -------------------------------------------------
# Data loaders
# -------------------------------------------------
def load_outputs():
    path = ROOT / "runs" / "output" / "baseline_outputs_v1.json"
    return json.loads(path.read_text())


def load_dataset():
    path = ROOT / "data" / "golden" / "dataset_v1.json"
    return json.loads(path.read_text())["samples"]


# -------------------------------------------------
# Evaluation helpers
# -------------------------------------------------
def aggregate_scores(results):
    totals = {}
    counts = {}

    for r in results:
        for metric, value in r["scores"].items():
            totals[metric] = totals.get(metric, 0.0) + value
            counts[metric] = counts.get(metric, 0) + 1

    return {
        metric: round(totals[metric] / counts[metric], 3)
        for metric in totals
    }


def evaluate():
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
        scores = {}

        for evaluator in evaluators:
            scores.update(
                evaluator.score(
                    input_text=sample["input"],
                    model_output=item["output"],
                    expected_output=sample["expected_output"],
                )
            )

        results.append(
            {
                "id": item["id"],
                "scores": scores,
            }
        )

    return results


# -------------------------------------------------
# Main execution
# -------------------------------------------------
def main():
    # ---------------- Evaluation ----------------
    results = evaluate()
    aggregate_metrics = aggregate_scores(results)

    # ---------------- Regression ----------------
    # NOTE: For now baseline == candidate (control test)
    baseline_metrics = aggregate_metrics
    candidate_metrics = aggregate_metrics

    detector = RegressionDetector(
        config_path=ROOT / "evals" / "regression_config.json"
    )

    regression = detector.compare(
        baseline=baseline_metrics,
        candidate=candidate_metrics,
    )

    # ---------------- Persist results ----------------
    results_path = ROOT / "runs" / "results" / "eval_results_v1.json"
    results_path.parent.mkdir(exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))

    # ---------------- MLflow Tracking ----------------
    logger = MLflowLogger(experiment_name="llm-eval-platform")
    logger.start_run(run_name="regression_check_v1")

    logger.log_metrics(candidate_metrics)

    logger.log_params(
        {
            "prompt_version": "v1",
            "dataset_version": "v1",
            "model": "llama3.1:8b",
            "regression_status": regression.status,
        }
    )

    logger.log_artifact(str(results_path))
    logger.end_run()

    # ---------------- Console output ----------------
    print("Aggregate metrics:", aggregate_metrics)
    print("Regression status:", regression.status)

    for detail in regression.details:
        print("⚠️", detail)


if __name__ == "__main__":
    main()
