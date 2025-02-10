from __future__ import annotations

from arcee.eval.evaluator import Evaluator

if __name__ == "__main__":
    # Initialise evaluator
    evaluator = Evaluator(
        model_id="gemma2:27b",
        model_type="ollama",
        data_path="data/RAMS/output/split_dev/dev.jsonlines",
        prompt_constructor_cls="DummyEAEPrompter",
        pattern_file_path="data/RAMS/patterns/pattern.json",
        batch_size=8,
    )

    # Evaluate!
    outputs = evaluator.evaluate()

    # Print
    print(outputs)
