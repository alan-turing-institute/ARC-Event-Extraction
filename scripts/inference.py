from __future__ import annotations

from arcee.eval.evaluator import Evaluator

if __name__ == "__main__":
    # Initialise evaluator
    evaluator = Evaluator(
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        model_type="hf",
        data_path="data/RAMS/output/splits/split1/test.jsonlines",
        prompt_constructor_cls="DummyEAEPrompter",
        pattern_file_path="data/RAMS/patterns/pattern.txt",
        dataset_name="rams",
        batch_size=8,
    )

    # Evaluate!
    outputs = evaluator.evaluate()

    # Print
    print(outputs)
