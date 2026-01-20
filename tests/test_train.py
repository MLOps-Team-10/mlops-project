from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import pytest


@pytest.fixture(autouse=True)
def disable_torch_compile(monkeypatch):
    """
    Automatically disables torch.compile for all tests in this module.
    It replaces torch.compile with a pass-through function.
    """
    monkeypatch.setattr(torch, "compile", lambda model, *args, **kwargs: model)


class DummyLoader:
    def __init__(self, batches: int = 2, batch_size: int = 4) -> None:
        self._batches = batches
        self._batch_size = batch_size
        self.dataset = [None] * (batches * batch_size)

    def __len__(self) -> int:
        return self._batches

    def __iter__(self):
        for _ in range(self._batches):
            images = torch.randn(self._batch_size, 3, 224, 224)
            labels = torch.zeros(self._batch_size, dtype=torch.long)
            yield images, labels


def test_train_saves_best_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import eurosat_classifier.train as train_mod
    import eurosat_classifier.model as model_mod

    class DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b = x.shape[0]
            # logits [B, 10]
            return torch.zeros(b, 10)

    # patch model constructor where it is defined
    monkeypatch.setattr(model_mod, "EuroSATModel", DummyModel)

    # fake dataloaders
    monkeypatch.setattr(
        train_mod,
        "get_dataloaders",
        lambda *a, **k: (DummyLoader(batches=2, batch_size=4), DummyLoader(batches=1, batch_size=4)),
    )

    # force validation improvement so save triggers
    monkeypatch.setattr(train_mod, "validate", lambda *a, **k: (0.123, 0.9))

    # avoid backprop complaining: make loss produce grad and model have params
    # easiest: patch optimizer + loss to no-op
    class DummyOptim:
        def __init__(self, params, lr):
            ...

        def zero_grad(self):
            ...

        def step(self):
            ...

    monkeypatch.setattr(train_mod.optim, "Adam", DummyOptim)

    class DummyLoss(torch.nn.Module):
        def forward(self, logits, labels):
            # return a tensor with grad
            return torch.tensor(0.0, requires_grad=True)

    monkeypatch.setattr(train_mod.nn, "CrossEntropyLoss", DummyLoss)

    saved = SimpleNamespace(called=False, path=None)

    def fake_save(obj, path):
        saved.called = True
        saved.path = Path(path)
        saved.path.parent.mkdir(parents=True, exist_ok=True)
        saved.path.write_bytes(b"dummy")

    monkeypatch.setattr(torch, "save", fake_save)

    train_mod.train(
        data_dir=str(tmp_path),  # not used due to fake_get_dataloaders
        batch_size=4,
        learning_rate=1e-3,
        epochs=1,
        num_workers=0,
        model_name="resnet18",
        log_interval=999,
        valid_fraction=0.2,
        num_classes=10,
        pretrained=False,
        logs_dir=tmp_path / "logs",
        models_dir=tmp_path / "models",
    )

    assert saved.called
    assert saved.path is not None
    assert saved.path.name == "eurosat_best.pth"
    assert saved.path.exists()
