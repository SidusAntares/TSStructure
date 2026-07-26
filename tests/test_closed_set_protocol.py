import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import dataset
from methods.structure_da import trainer as structure_da_trainer

import train


def _write_metadata(tmp_path, labels):
    dataset_root = tmp_path / "france" / "tile" / "2017"
    (dataset_root / "data").mkdir(parents=True)
    (dataset_root / "meta").mkdir()
    metadata = {
        "dates": [20170101],
        "start_date": 20170101,
        "parcels": [
            {"label": label, "n_pixels": 1, "geometric_features": [0.0]}
            for label in labels
        ],
    }
    with open(dataset_root / "meta" / "metadata.pkl", "wb") as handle:
        pickle.dump(metadata, handle)


def test_closed_set_filters_unretained_and_unmapped_classes(tmp_path, monkeypatch):
    _write_metadata(tmp_path, [10, 20, 999])
    monkeypatch.setattr(
        dataset.label_utils,
        "get_code_to_class",
        lambda country, combine_spring_and_winter=False: {
            10: "kept", 20: "not_kept"
        },
    )

    ds = dataset.PixelSetData(
        str(tmp_path), "france/tile/2017", ["kept"], closed_set=True
    )

    assert ds.get_labels().tolist() == [0]
    assert [sample[1] for sample in ds.samples] == [0]


def test_open_set_maps_unretained_and_unmapped_classes_to_unknown(
    tmp_path, monkeypatch
):
    _write_metadata(tmp_path, [10, 20, 999])
    monkeypatch.setattr(
        dataset.label_utils,
        "get_code_to_class",
        lambda country, combine_spring_and_winter=False: {
            10: "kept", 20: "not_kept"
        },
    )

    ds = dataset.PixelSetData(
        str(tmp_path),
        "france/tile/2017",
        ["kept", "unknown"],
        closed_set=False,
    )

    assert ds.get_labels().tolist() == [0, 1, 1]


def test_closed_set_exposes_original_parcel_ids(tmp_path, monkeypatch):
    _write_metadata(tmp_path, [20, 10, 20, 20, 10, 10])
    monkeypatch.setattr(
        dataset.label_utils,
        "get_code_to_class",
        lambda country, combine_spring_and_winter=False: {
            10: "kept", 20: "not_kept"
        },
    )

    ds = dataset.PixelSetData(
        str(tmp_path), "france/tile/2017", ["kept"], closed_set=True
    )

    assert ds.get_parcel_indices().tolist() == [1, 4, 5]


def test_closed_set_uses_combined_spring_winter_mapping(tmp_path, monkeypatch):
    _write_metadata(tmp_path, [10])

    def get_code_to_class(country, combine_spring_and_winter=False):
        if combine_spring_and_winter:
            return {10: "wheat"}
        return {10: "spring_wheat"}

    monkeypatch.setattr(
        dataset.label_utils, "get_code_to_class", get_code_to_class
    )

    ds = dataset.PixelSetData(
        str(tmp_path),
        "france/tile/2017",
        ["wheat"],
        closed_set=True,
        combine_spring_and_winter=True,
    )

    assert ds.get_labels().tolist() == [0]


def test_split_accepts_explicit_original_parcel_ids():
    eligible = {
        "source": [1, 4, 5, 8, 10],
        "target": [2, 7, 9, 12, 20],
    }

    splits = train.create_train_val_test_folds(
        ["source", "target"], 1, eligible, val_ratio=0.2, test_ratio=0.2
    )[0]

    for name, expected in eligible.items():
        train_ids = splits[name]["train"]
        val_ids = splits[name]["val"]
        test_ids = splits[name]["test"]
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
        assert train_ids | val_ids | test_ids == set(expected)


class _ProtocolDataset:
    calls = []

    def __init__(
        self,
        data_root,
        dataset_name,
        classes,
        transform=None,
        indices=None,
        with_extra=False,
        closed_set=False,
        combine_spring_and_winter=False,
    ):
        self.dataset_name = dataset_name
        self.classes = classes
        self.closed_set = closed_set
        self.calls.append(
            (
                dataset_name,
                tuple(classes),
                closed_set,
                combine_spring_and_winter,
            )
        )

    def get_labels(self):
        source_country = self.dataset_name.split("/")[0]
        if source_country == "france" and self.classes == ["alpha", "beta"]:
            return np.array([0] * 201 + [1] * 199)
        if source_country == "denmark" and self.classes == ["beta", "alpha"]:
            return np.array([0] * 205 + [1] * 50)
        return np.array([], dtype=np.int64)

    def get_parcel_indices(self):
        if self.dataset_name.startswith("france/"):
            return np.array([1, 4, 5], dtype=np.int64)
        return np.array([2], dtype=np.int64)


def _protocol_config(tmp_path, source, target):
    return SimpleNamespace(
        closed_set=True,
        data_root="unused",
        source=source,
        target=target,
        combine_spring_and_winter=False,
        output_dir=str(tmp_path),
        seed=1,
        val_ratio=0.1,
        test_ratio=0.2,
    )


def test_closed_set_classes_are_source_defined_and_protocol_is_recorded(
    tmp_path, monkeypatch
):
    _ProtocolDataset.calls = []
    monkeypatch.setattr(train, "PixelSetData", _ProtocolDataset)
    monkeypatch.setattr(
        train.label_utils,
        "get_classes",
        lambda country, combine_spring_and_winter=False: (
            ["alpha", "unknown", "beta"]
            if country == "france"
            else ["beta", "unknown", "alpha"]
        ),
    )
    config = _protocol_config(
        tmp_path, "france/FR1/2017", "denmark/DK1/2017"
    )

    eligible, protocol = train.prepare_data_protocol(config)

    assert config.classes == ["alpha"]
    assert config.num_classes == 1
    assert eligible == {
        "france/FR1/2017": [1, 4, 5],
        "denmark/DK1/2017": [2],
    }
    assert protocol["classes"] == ["alpha"]
    assert protocol["source_class_counts"] == {"alpha": 201}
    assert protocol["class_to_idx"] == {"alpha": 0}
    assert all(call[2] for call in _ProtocolDataset.calls)
    saved = json.loads((tmp_path / "closed_set_protocol.json").read_text())
    assert saved == protocol


def test_task_local_classes_stay_same_for_same_source_and_may_change_by_source(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(train, "PixelSetData", _ProtocolDataset)
    monkeypatch.setattr(
        train.label_utils,
        "get_classes",
        lambda country, combine_spring_and_winter=False: (
            ["alpha", "unknown", "beta"]
            if country == "france"
            else ["beta", "unknown", "alpha"]
        ),
    )

    first = _protocol_config(tmp_path, "france/FR1/2017", "denmark/DK1/2017")
    second = _protocol_config(tmp_path, "france/FR1/2017", "austria/AT1/2017")
    reverse = _protocol_config(tmp_path, "denmark/DK1/2017", "france/FR1/2017")
    train.prepare_data_protocol(first)
    train.prepare_data_protocol(second)
    train.prepare_data_protocol(reverse)

    assert first.classes == second.classes == ["alpha"]
    assert reverse.classes == ["beta"]


def test_open_set_protocol_preserves_unknown_and_length_based_indices(
    tmp_path, monkeypatch
):
    calls = []

    class OpenDataset:
        def __init__(
            self,
            data_root,
            dataset_name,
            classes,
            combine_spring_and_winter=False,
        ):
            calls.append(
                (dataset_name, tuple(classes), combine_spring_and_winter)
            )
            self.dataset_name = dataset_name

        def get_labels(self):
            return np.array([0] * 200 + [1] * 250 + [2] * 199)

        def __len__(self):
            return 649 if self.dataset_name == "source" else 17

    monkeypatch.setattr(train, "PixelSetData", OpenDataset)
    monkeypatch.setattr(
        train.label_utils,
        "get_classes",
        lambda country, combine_spring_and_winter=False: [
            "alpha",
            "unknown",
            "beta",
        ],
    )
    config = _protocol_config(tmp_path, "source", "target")
    config.closed_set = False
    config.combine_spring_and_winter = True

    indices, protocol = train.prepare_data_protocol(config)

    assert config.classes == ["alpha", "unknown"]
    assert indices == {"source": 649, "target": 17}
    assert protocol is None
    assert calls == [
        ("source", ("alpha", "unknown", "beta"), True),
        ("target", ("alpha", "unknown"), True),
    ]


class _LoaderDataset:
    calls = []

    def __init__(self, *args, **kwargs):
        self.calls.append(
            (
                kwargs.get("closed_set"),
                kwargs.get("combine_spring_and_winter"),
            )
        )
        self.transform = kwargs.get("transform")

    def __len__(self):
        return 2

    def get_labels(self):
        return np.array([0, 0])

    def get_shapes(self):
        return [(1, 10, 1), (1, 10, 1)]


class _Loader:
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset

    def __len__(self):
        return 1


def test_closed_set_is_propagated_to_structure_da_and_evaluation_loaders(
    monkeypatch
):
    config = SimpleNamespace(
        closed_set=True,
        data_root="data",
        source="source",
        target="target",
        classes=["crop"],
        model="pseltae",
        num_pixels=2,
        seq_length=3,
        with_shift_aug=False,
        max_shift_aug=5,
        shift_aug_p=1.0,
        num_workers=0,
        batch_size=2,
        lr=0.001,
        weight_decay=0.0,
        train_on_target=False,
        combine_spring_and_winter=True,
        with_extra=False,
    )
    splits = {
        "source": {"train": {1}, "val": {2}, "test": {3}},
        "target": {"train": {4}, "val": {5}, "test": {6}},
    }

    _LoaderDataset.calls = []
    monkeypatch.setattr(structure_da_trainer, "PixelSetData", _LoaderDataset)
    monkeypatch.setattr(
        structure_da_trainer,
        "create_train_loader",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
    )
    with pytest.raises(RuntimeError, match="stop"):
        structure_da_trainer.create_structure_da_train_loaders(config, splits)
    assert _LoaderDataset.calls == [(True, True), (True, True)]

    _LoaderDataset.calls = []
    monkeypatch.setattr(dataset, "PixelSetData", _LoaderDataset)
    monkeypatch.setattr(dataset.data, "DataLoader", _Loader)
    monkeypatch.setattr(dataset, "GroupByShapesBatchSampler", lambda *args, **kwargs: [])
    dataset.create_evaluation_loaders("target", splits, config)
    assert _LoaderDataset.calls == [(True, True), (True, True)]
