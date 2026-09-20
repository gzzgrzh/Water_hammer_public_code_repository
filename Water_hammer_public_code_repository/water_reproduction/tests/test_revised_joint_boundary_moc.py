import numpy as np
import pytest

from water16_reproduction.revised_joint_boundary_moc import classification_counts, metric_relative_error, save_case


def test_metric_relative_error_is_reference_normalized() -> None:
    assert metric_relative_error(100.0, 95.0) == 0.05
    assert metric_relative_error(-100.0, -105.0) == 0.05


def test_classification_counts_exposes_false_safe_cases() -> None:
    rows = [
        {"registered_safe": True, "moc_safe": True},
        {"registered_safe": True, "moc_safe": False},
        {"registered_safe": False, "moc_safe": True},
        {"registered_safe": False, "moc_safe": False},
    ]
    assert classification_counts(rows) == {
        "true_safe": 1,
        "false_safe": 1,
        "false_unsafe": 1,
        "true_unsafe": 1,
    }


def test_case_checkpoint_appears_only_as_complete_directory(tmp_path) -> None:
    z = np.linspace(0.0, 1.0, 5)
    t = np.linspace(0.0, 1.0, 7)
    field = np.zeros((len(z), len(t)))
    truth = {"z": z, "t": t, "V": field, "uz": field, "P": field, "sigma_z": field}
    case_dir = tmp_path / "cases" / "boundary_001"
    save_case(case_dir, {"case_id": "boundary_001"}, truth, {"comparison_row": {"case_id": "boundary_001"}}, 3)
    assert (case_dir / "truth_evaluation_grid.npz").exists()
    assert (case_dir / "result.json").exists()
    assert not list(case_dir.parent.glob(".partial_*"))
    with pytest.raises(FileExistsError):
        save_case(case_dir, {"case_id": "boundary_001"}, truth, {"comparison_row": {"case_id": "boundary_001"}}, 3)
