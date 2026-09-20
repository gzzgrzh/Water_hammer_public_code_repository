import json

import pytest

from water16_reproduction.parametric_fssi_formal_release import acceptance, test_release as run_test_release


def test_formal_acceptance_uses_all_four_registered_metrics() -> None:
    config = {
        "acceptance": {
            "formal_pressure_nrmse": 0.02,
            "formal_axial_stress_nrmse": 0.03,
            "formal_pressure_peak_relative_error": 0.03,
            "formal_stress_peak_relative_error": 0.05,
        }
    }
    maximum = {
        "P_nrmse": 0.019,
        "sigma_z_nrmse": 0.029,
        "pressure_peak_relative_error": 0.029,
        "stress_peak_relative_error": 0.049,
    }
    assert acceptance({"maximum_by_metric": maximum}, config)["status"] == "pass"
    for key in maximum:
        failed = dict(maximum)
        failed[key] = 0.2
        assert acceptance({"maximum_by_metric": failed}, config)["status"] == "failed"


def test_test_split_remains_sealed_without_validation_release(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(RuntimeError, match="remains sealed"):
        run_test_release(tmp_path / "config.json", checkpoint, tmp_path / "output", "cpu")


def test_test_split_remains_sealed_after_checkpoint_change(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"new checkpoint")
    release_dir = tmp_path / "output" / "validation_release"
    release_dir.mkdir(parents=True)
    (release_dir / "release_report.json").write_text(
        json.dumps({"status": "pass", "checkpoint_sha256": "old"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        run_test_release(tmp_path / "config.json", checkpoint, tmp_path / "output", "cpu")
