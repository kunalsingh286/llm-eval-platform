import json
from evals.base import BaseEvaluator


class FormatAccuracyEvaluator(BaseEvaluator):
    name = "format_accuracy"

    def score(self, input_text, model_output, expected_output):
        """
        Validates structured output.
        Currently supports JSON object comparison.
        """

        if not isinstance(expected_output, dict):
            # Not applicable
            return {"format_accuracy": 1.0}

        try:
            parsed = json.loads(model_output)
        except Exception:
            return {"format_accuracy": 0.0}

        for key in expected_output.keys():
            if key not in parsed:
                return {"format_accuracy": 0.0}

        return {"format_accuracy": 1.0}
