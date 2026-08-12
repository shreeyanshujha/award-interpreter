import pytest

from knee_mri.config import Config


def test_defaults_are_valid():
    cfg = Config()
    assert cfg.backbone == "resnet18"
    assert cfg.tasks == ["abnormal", "acl", "meniscus"]


def test_overrides_skip_none():
    cfg = Config.from_dict({"epochs": 3}, epochs=None, lr=0.5)
    assert cfg.epochs == 3   # None override ignored
    assert cfg.lr == 0.5     # real override applied


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        Config.from_dict({"not_a_field": 1})


def test_bad_enum_rejected():
    with pytest.raises(ValueError):
        Config(pool="median")
    with pytest.raises(ValueError):
        Config(backbone="vgg")


def test_roundtrip_dict():
    cfg = Config(epochs=7)
    assert Config.from_dict(cfg.to_dict()).epochs == 7
