import copy

from water16_reproduction.parametric_fssi_forward import (
    DEFAULT_CONFIG,
    PARAMETER_NAMES,
    audit_config,
    load_config,
    materialize_cases,
)


def test_frozen_forward_design_has_registered_counts_and_no_overlap() -> None:
    config = load_config(DEFAULT_CONFIG)
    report = audit_config(config)
    assert report["status"] == "pass", report["issues"]
    assert report["stages"]["pilot"]["split_counts"] == {
        "train": 12,
        "validation": 4,
        "test": 6,
    }
    assert report["stages"]["formal"]["split_counts"] == {
        "train": 72,
        "validation": 12,
        "test": 24,
    }
    assert report["stages"]["formal"]["unique_parameter_points"] == 108


def test_formal_design_is_deterministic_and_respects_transforms() -> None:
    config = load_config(DEFAULT_CONFIG)
    first = materialize_cases(config, "formal")
    second = materialize_cases(config, "formal")
    assert first == second
    assert len(first) == 108
    for case in first:
        for name in PARAMETER_NAMES:
            definition = config["parameter_domain"][name]
            assert definition["lower"] <= case[name] <= definition["upper"]


def test_validation_data_are_forbidden_from_training() -> None:
    config = load_config(DEFAULT_CONFIG)
    forbidden = set(config["training_data_policy"]["forbidden"])
    assert "MOC full-field values" in forbidden
    assert "validation cases" in forbidden
    assert "test cases" in forbidden
    assert config["parameter_analysis"]["start_condition"] == (
        "classic and held-out numerical validation passed"
    )


def test_audit_rejects_a_duplicate_parameter_point() -> None:
    config = copy.deepcopy(load_config(DEFAULT_CONFIG))
    cases = config["case_design"]["pilot"]["cases"]
    for name in PARAMETER_NAMES:
        cases[1][name] = cases[0][name]
    report = audit_config(config)
    assert report["status"] == "failed"
    assert "pilot: duplicate parameter point across splits" in report["issues"]
