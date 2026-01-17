from evals.base import BaseEvaluator


class RelevanceEvaluator(BaseEvaluator):
    name = "relevance"

    def score(self, input_text, model_output, expected_output):
        """
        Relevance applies only to text-based expected outputs.
        Structured outputs are treated as not applicable.
        """

        if not isinstance(expected_output, str):
            return {"relevance": 1.0}

        if not isinstance(model_output, str):
            return {"relevance": 0.0}

        input_tokens = set(input_text.lower().split())
        output_tokens = set(model_output.lower().split())

        if not input_tokens:
            return {"relevance": 0.0}

        overlap = input_tokens.intersection(output_tokens)
        score = len(overlap) / len(input_tokens)

        return {"relevance": round(min(score, 1.0), 3)}
