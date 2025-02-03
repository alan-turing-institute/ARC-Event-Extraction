from __future__ import annotations

import pytest

from from_textee import data_prep


@pytest.mark.usefixtures("rams_data_paths")
def test_read_docs(testing_rams_dir: pytest.fixture) -> None:
    all_objs = data_prep.read_docs(testing_rams_dir)

    assert len(all_objs) == 10
    assert isinstance(all_objs, list)
    assert isinstance(all_objs[0], dict)
