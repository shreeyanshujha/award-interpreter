from pathlib import Path

from knee_mri.config import Config
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
