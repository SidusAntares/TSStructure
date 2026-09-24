import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_structure_vs_timematch.py"
SPEC = importlib.util.spec_from_file_location("structure_vs_timematch", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_representation_metrics_domain_inter_and_intra():
    source = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    target = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    labels = np.array([0, 0, 1, 1])
    result, per_class = MODULE.representation_metrics(source, labels, target, labels)
    assert np.isclose(result["domain_gap"], 0.)
    assert np.isclose(result["source_inter"], 1.)
    assert np.isclose(result["target_inter"], 1.)
    assert np.isclose(result["source_intra"], 0.)
    assert np.isclose(result["target_intra"], 0.)
    assert all(np.isclose(row["domain_gap"], 0.) for row in per_class.values())


def test_pseudo_metrics_coverage_precision_macro_f1_and_js():
    truth = np.array([0, 0, 1, 1])
    prediction = np.array([0, 1, 1, 0])
    accepted = np.array([True, True, True, False])
    result, per_class = MODULE.pseudo_label_metrics(truth, prediction, accepted, 2)
    assert np.isclose(result["coverage"], .75)
    assert np.isclose(result["precision"], 2 / 3)
    assert np.isclose(result["macro_f1"], 2 / 3)
    assert result["distribution_js"] > 0
    assert per_class[0]["gt_count"] == 2
    assert per_class[1]["accepted_count"] == 1


def test_prototype_margin_sign_and_nearest_prediction():
    features = np.array([[1., 0.], [0., 1.], [1., 0.]])
    labels = np.array([0, 1, 1])
    prototypes = np.eye(2)
    result, per_class = MODULE.prototype_geometry(features, labels, prototypes)
    assert result["margin_mean"] > -1
    assert np.isclose(result["positive_fraction"], 2 / 3)
    assert np.isclose(result["accuracy"], 2 / 3)
    assert per_class[0]["nearest_accuracy"] == 1.
    assert per_class[1]["margin_mean"] < 1.


def test_anchor_usage():
    responses = np.array([[3., 1., 0.], [2., 1., 0.], [0., 4., 1.], [0., 5., 1.]])
    result = MODULE.anchor_usage(responses)
    assert result["active_anchor_count"] == 2
    assert np.isclose(result["entropy"], np.log(2.))
    assert np.isclose(result["max_fraction"], .5)


def test_query_ratio_summary():
    master = np.array([[3., 4.], [0., 2.]])
    shape = np.array([[0., 5.], [0., 1.]])
    result = MODULE.query_ratio_summary(master, shape)
    assert np.allclose(result["values"], [1., .5])
    assert np.isclose(result["mean"], .75)
    assert np.isclose(result["p50"], .75)


def test_attention_js_identical_and_different():
    first = np.array([[[.5, .5]], [[.9, .1]]])
    assert np.isclose(MODULE.attention_js(first, first), 0.)
    second = np.array([[[.9, .1]], [[.1, .9]]])
    assert MODULE.attention_js(first, second) > 0.


def test_final_student_checkpoint_role_validation():
    packet = {
        "epoch": 19,
        "config": {"epochs": 20, "output_student": True},
        "teacher_state_dict": {},
    }
    MODULE.validate_final_student_packet("checkpoint_last.pt", packet)
    for name, bad in (
        ("model.pt", packet),
        ("checkpoint_last.pt", {"epoch": 18, "config": packet["config"]}),
        ("checkpoint_last.pt", {"epoch": 19, "config": {"epochs": 20, "output_student": False}}),
    ):
        try:
            MODULE.validate_final_student_packet(name, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid final-student checkpoint was accepted")
