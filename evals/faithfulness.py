from evals.base import BaseEvaluator


class FaithfulnessEvaluator(BaseEvaluator):
    name = "faithfulness"

    def score(self, input_text, model_output, expected_output):
        """
        Faithfulness applies only to text-based expected outputs.
        Structured outputs are treated as not applicable.
        """

        if not isinstance(expected_output, str):
            return {"faithfulness": 1.0}

        if not isinstance(model_output, str):
            return {"faithfulness": 0.0}

        expected_tokens = set(expected_output.lower().split())
        output_tokens = set(model_output.lower().split())

        if not expected_tokens:
            return {"faithfulness": 0.0}

        overlap = expected_tokens.intersection(output_tokens)
        score = len(overlap) / len(expected_tokens)

        return {"faithfulness": round(min(score, 1.0), 3)}
