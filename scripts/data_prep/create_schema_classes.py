from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path

from arcee.dataset.load import load_schema
from jinja2 import Template
from pydantic import BaseModel

"""
This script takes the pattern file from TextEE (for the BARTGen model) and creates a
Python file with the schema defined as Python classes.
"""


def render_schema(pattern_file: str, output_file: str) -> None:
    template = """from pydantic import BaseModel

{% for event_type in schema %}
class {{event_type}}(BaseModel):{% for key, value in schema[event_type].items() %}
    {{key}} : {{value}}{% endfor %}
{% endfor %}

    """

    j2_template = Template(template)
    schema = load_schema(pattern_file)
    output = j2_template.render(schema=schema)

    with open(output_file, mode="w", encoding="utf-8") as f:
        f.write(output)

    print(f"Rendered output saved to {output_file}")


def load_module_from_path(file_path: str):
    """Dynamically loads a Python file as a module."""
    module_name = Path(file_path).stem  # Extract filename without extension
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def schema_to_json_dict(path_name: str, output_file: str) -> None:
    """
    Extracts JSON schemas from Pydantic classes in a module and saves them as a
    JSON file.
    """
    module = load_module_from_path(path_name)
    schemas = {}

    # Iterate over all objects in the module
    for name, obj in inspect.getmembers(module):
        if (
            inspect.isclass(obj)
            and issubclass(obj, BaseModel)
            and obj.__module__ == module.__name__
        ):
            schemas[name] = obj.model_json_schema()  # Generate JSON schema

    # Save the schemas to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(schemas, f, indent=4, ensure_ascii=False)

    print(f"Schema saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Render schema from a pattern JSON file."
    )
    parser.add_argument(
        "--pattern_file", type=str, help="Path to the pattern JSON file"
    )
    parser.add_argument(
        "--class_output_file", type=str, help="Path to save the rendered class output"
    )
    parser.add_argument(
        "--json_output_file",
        type=str,
        help="Path to save the rendered json dict output",
    )

    args = parser.parse_args()
    render_schema(args.pattern_file, args.class_output_file)

    schema_to_json_dict(args.class_output_file, args.json_output_file)


if __name__ == "__main__":
    main()
