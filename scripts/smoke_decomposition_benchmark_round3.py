import argparse
import importlib.util

import torch

from models.decomposition_benchmark import (
    CeemdanSeDecompositionClassifier,
    VmdDecompositionClassifier,
)


def require(module_name, package_name):
    if importlib.util.find_spec(module_name) is None:
        raise SystemExit(
            "Missing dependency {}. Install offline package '{}' first.".format(
                module_name, package_name
            )
        )


def make_batch(batch_size, length, channels, num_pixels):
    torch.manual_seed(1)
    pixels = torch.randn(batch_size, length, channels, num_pixels)
    mask = torch.ones(batch_size, length, num_pixels)
    positions = torch.arange(length).unsqueeze(0).repeat(batch_size, 1)
    extra = torch.zeros(batch_size, 4)
    return pixels, mask, positions, extra


def run_model(model, batch):
    pixels, mask, positions, extra = batch
    sample = {
        "pixels": pixels,
        "valid_pixels": mask,
        "positions": positions,
        "extra": extra,
    }
    model.fit_component_normalizers([sample], device="cpu")
    model.eval()
    with torch.no_grad():
        logits = model(pixels, mask, positions, extra)
    print(type(model).__name__, tuple(logits.shape))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ceemdan_trials", type=int, default=2)
    args = parser.parse_args()

    require("vmdpy", "vmdpy")
    require("PyEMD", "EMD-signal")

    batch = make_batch(batch_size=1, length=30, channels=2, num_pixels=2)
    common = dict(
        input_dim=2,
        mlp1=[2, 4],
        pooling="mean_std",
        mlp2=[8, 8],
        with_extra=False,
        n_head=2,
        d_k=2,
        d_model=8,
        mlp3=[8, 4],
        dropout=0.0,
        mlp4=[4],
        num_classes=3,
        max_temporal_shift=10,
    )
    run_model(VmdDecompositionClassifier(num_modes=3, **common), batch)
    run_model(
        CeemdanSeDecompositionClassifier(
            trials=args.ceemdan_trials, noise_seed=1, **common
        ),
        batch,
    )


if __name__ == "__main__":
    main()
