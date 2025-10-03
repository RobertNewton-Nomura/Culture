import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "Culture_Explorer.ipynb"


def _execute_notebook() -> Dict[str, Any]:
    with NOTEBOOK_PATH.open() as handle:
        nb = json.load(handle)
    env: Dict[str, Any] = {"__name__": "__notebook_tests__"}
    for index, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if not source.strip():
            continue
        stripped = source.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            continue
        code = compile(source, filename=f"notebook_cell_{index}", mode="exec")
        exec(code, env)
    return env


@pytest.fixture(scope="session")
def notebook_env() -> Dict[str, Any]:
    return _execute_notebook()


def test_classes_and_helpers_exposed(notebook_env: Dict[str, Any]) -> None:
    for name in [
        "CulturalDataset",
        "GroupManager",
        "OpenAIInsightGenerator",
        "CultureExplorerService",
        "render_static_culture_map",
    ]:
        assert name in notebook_env, f"Expected {name} to be defined in the notebook."


def test_dataset_loaded(notebook_env: Dict[str, Any]) -> None:
    dataset = notebook_env["dataset"]
    countries = dataset.get_countries()
    years = dataset.get_years()
    assert countries, "At least one country should be available."
    assert years == sorted(years), "Years should be sorted ascending."
    matrix = dataset.get_question_matrix(countries[:2], years[-1])
    assert isinstance(matrix, pd.DataFrame)
    assert not matrix.empty


def test_group_manager_profiles(notebook_env: Dict[str, Any]) -> None:
    dataset = notebook_env["dataset"]
    manager = notebook_env["group_manager"]
    groups = dataset.get_group_questions()
    sample_group = next(iter(groups))
    sample_question = groups[sample_group][0]
    manager.add_member(
        "Notebook QA",
        "Analyst",
        {(sample_group, sample_question): 0.65},
        reference_year=dataset.get_years()[-1],
    )
    profile = manager.compute_group_profile("Notebook QA")
    assert profile[(sample_group, sample_question)] == pytest.approx(0.65)
    match = manager.match_closest_country("Notebook QA")
    assert not match.empty


def test_service_interface(notebook_env: Dict[str, Any]) -> None:
    service = notebook_env["service"]
    assert len(service.get_countries()) == len(set(service.get_countries()))
    latest_year = max(service.get_years())
    sample_countries = service.get_countries()[:3]
    score_matrix = notebook_env["dataset"].get_question_matrix(sample_countries, latest_year)
    assert isinstance(score_matrix, pd.DataFrame)
    assert score_matrix.columns.tolist() == sample_countries
    weights = {
        (group, question): 1.0
        for group, questions in service.get_group_questions().items()
        for question in questions
    }
    weighted = service.dataset.compute_weighted_group_scores(sample_countries, latest_year, weights)
    assert set(weighted.index) == set(sample_countries)
