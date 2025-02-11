from __future__ import annotations

import argparse

from arcee.dataset.load import load_schema
from jinja2 import Template

"""
This script takes the pattern file from TextEE (for the BARTGen model) and creates a
Python file with the schema defined as Python classes.
"""


def render_schema(pattern_file: str, output_file: str):
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


def main():
    parser = argparse.ArgumentParser(
        description="Render schema from a pattern JSON file."
    )
    parser.add_argument(
        "--pattern_file", type=str, help="Path to the pattern JSON file"
    )
    parser.add_argument(
        "--output_file", type=str, help="Path to save the rendered output"
    )

    args = parser.parse_args()
    render_schema(args.pattern_file, args.output_file)


if __name__ == "__main__":
    main()
