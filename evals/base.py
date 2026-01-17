from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseEvaluator(ABC):
    name: str

    @abstractmethod
    def score(
        self,
        input_text: str,
        model_output: Any,
        expected_output: Any,
    ) -> Dict[str, float]:
        """
        Returns a dict of metric_name -> score (0.0 to 1.0)
        """
        pass
