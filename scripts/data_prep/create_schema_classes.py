from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
from types import ModuleType

from arcee.dataset.load import pattern_to_class_schema
from jinja2 import Template
from pydantic import BaseModel

"""
This script takes the pattern file from TextEE (for the BARTGen model) and creates a
Python file with the schema defined as Python classes.
"""


def render_schema(pattern_file: str, output_file: str) -> None:
    """
    This uses a template to convert each event type in the pattern file into a class
    and saves the output in a Python script.

    Parameters
    ----------
    pattern_file: str
        The file path where the pattern file is stored for the dataset, this should
        include the file name and extension, expected to be JSON format
        e.g. "data/RAMS/patterns/pattern.json"
    output_file: str
        The file path where the output is to be stored for the dataset, this should
        include the file name and extension, expected to be JSON format
        e.g. "data/RAMS/patterns/schema.json"
    """

    # Create template
    template = """from pydantic import BaseModel

{% for event_type in schema %}
class {{event_type}}(BaseModel):{% for key, value in schema[event_type].items() %}
    {{key}} : {{value}}{% endfor %}
{% endfor %}

    """
    j2_template = Template(template)

    # Load the pattern file to be used as a schema (this converts the names
    # of the event types to valid class names and defines the parameters
    # of a class as the argument roles)
    schema = pattern_to_class_schema(pattern_file)

    # Apply the template to the schema
    output = j2_template.render(schema=schema)

    # Save output
    with open(output_file, mode="w", encoding="utf-8") as f:
        f.write(output)

    print(f"Rendered output saved to {output_file}")


def load_module_from_path(file_path: str) -> ModuleType:
    """
    Dynamically loads a Python file as a module.

    Parameters
    ----------
    file_path : str
        The path to the Python file to be loaded as a module.

    Returns
    -------
    types.ModuleType
        The loaded module object, allowing access to its attributes and definitions.

    """
    module_name = Path(file_path).stem  # Extract filename without extension
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def schema_to_json_dict(path_name: str, output_file: str) -> None:
    """
    Extracts JSON schemas from Pydantic classes in a Python file and saves them as a
    JSON file.

    Parameters
    ----------
    path_name : str
        The file path of the Python script containing Pydantic model definitions.
    output_file : str
        The file path where the extracted JSON schemas should be saved.

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
