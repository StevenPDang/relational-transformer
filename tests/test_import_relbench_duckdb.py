import importlib.util
from pathlib import Path

import pytest


_MODULE_SPEC = importlib.util.spec_from_file_location(
    "import_relbench_duckdb_under_test",
    Path(__file__).parents[1] / "scripts/import_relbench_duckdb.py",
)
import_relbench_duckdb = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(import_relbench_duckdb)
organized_output_path = import_relbench_duckdb.organized_output_path


def test_organized_output_path_uses_dataset_name():
    assert organized_output_path("rel-f1", Path("data/duckdb")) == Path(
        "data/duckdb/rel-f1.duckdb"
    )


@pytest.mark.parametrize("name", ["", "../rel-f1", "nested/rel-f1"])
def test_organized_output_path_rejects_unsafe_names(name):
    with pytest.raises(ValueError, match="Unsafe dataset name"):
        organized_output_path(name, Path("data/duckdb"))
