from pathlib import Path

from knee_mri.config import Config
from knee_mri.data import RSNA_TASKS, generate_rsna_dataset
from knee_mri.engine import train


def test_training_runs_and_writes_checkpoints(synthetic_root, tmp_path):
    cfg = Config(
        data_root=str(synthetic_root),
        backbone="alexnet",   # smaller/faster than resnet for the smoke test
        image_size=64,        # AlexNet needs >= ~63px input; 64 is the safe minimum
        epochs=1,
        lr=1e-4,
        device="cpu",
        out_dir=str(tmp_path / "runs"),
        log_every=100,
    )
    result = train(cfg, run_name="smoke")

    out = Path(result["out_dir"])
    assert (out / "last.pt").is_file()
    assert (out / "config.json").is_file()
    assert (out / "history.json").is_file()
    assert len(result["history"]) == 1
    # A finite loss means the forward/backward pass wired up correctly.
    assert result["history"][0]["train_loss"] == result["history"][0]["train_loss"]


def test_rsna_training_smoke(tmp_path):
    root = tmp_path / "rsna"
    generate_rsna_dataset(out=root, n_train=8, n_valid=4, gold_fraction=0.25,
                          size=32, slices=(6, 10), seed=0)
    cfg = Config(
        data_root=str(root),
        dataset_format="rsna",
        tasks=list(RSNA_TASKS),
        backbone="alexnet",
        image_size=64,
        input_mode="triplets",
        positions_per_plane=3,
        epochs=1,
        lr=1e-4,
        device="cpu",
        out_dir=str(tmp_path / "runs"),
        log_every=100,
    )
    result = train(cfg, run_name="rsna-smoke")
    out = Path(result["out_dir"])
    assert (out / "last.pt").is_file()
    # 12-task head + weighted loss produced a finite training loss.
    loss = result["history"][0]["train_loss"]
    assert loss == loss and loss > 0
