import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class RagasEvaluator:
    """Automated evaluation harness for measuring Faithfulness, Relevance, and Precision."""

    def __init__(self):
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            self.evaluate_fn = evaluate
            self.metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]
            logger.info("Ragas evaluation harness initialized.")
        except ImportError:
            logger.warning("Ragas is not installed. Install via `pip install ragas`.")
            self.evaluate_fn = None

    def run_benchmark(self, dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Runs Ragas evaluation over a list of test cases:
        dataset = [
            {
                "question": "What is ...?",
                "contexts": ["Chunk 1 text...", "Chunk 2 text..."],
                "answer": "Generated answer...",
                "ground_truth": "Expected human reference..."
            }
        ]
        """
        if not self.evaluate_fn:
            logger.error("Cannot run benchmark: Ragas not installed.")
            return {}

        from datasets import Dataset
        ragas_dataset = Dataset.from_list(dataset)
        results = self.evaluate_fn(ragas_dataset, metrics=self.metrics)
        return results
