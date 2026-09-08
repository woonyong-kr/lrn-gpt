import torch
import pytest
from src.runnable import TrainingRun

CONFIG = {"context_length": 8, "emb_dim": 16, "n_heads": 2, "n_layers": 1, "drop_rate": 0.2, "vocab_size": 270, "batch_size": 3, "seed": 41, "learning_rate": 0.001}
TEXT = "작은 언어 모델은 다음 토큰을 예측한다. A small model learns tokens.\n" * 20


def test_resumed_training_matches_uninterrupted_with_dropout(tmp_path):
    full = TrainingRun.create(TEXT, TEXT, CONFIG)
    full.step()
    full.step()
    full.step()
    expected = {k: v.clone() for k, v in full.model.state_dict().items()}
    split = TrainingRun.create(TEXT, TEXT, CONFIG)
    split.step()
    split.save(tmp_path / "state.pt")
    resumed = TrainingRun.load(tmp_path / "state.pt", TEXT, TEXT)
    assert resumed.tokenizer.decode(resumed.tokenizer.encode("처음 보는 🙂 byte")) == "처음 보는 🙂 byte"
    resumed.step()
    resumed.step()
    assert resumed.global_step == 3
    for k, v in resumed.model.state_dict().items():
        torch.testing.assert_close(v, expected[k], rtol=0, atol=0)


def test_changed_corpus_is_rejected(tmp_path):
    run = TrainingRun.create(TEXT, TEXT, CONFIG)
    run.save(tmp_path / "state.pt")
    with pytest.raises(ValueError, match="corpus"):
        TrainingRun.load(tmp_path / "state.pt", TEXT + "modified", TEXT)


def test_model_blocks_future_tokens_and_matches_independent_loss():
    run = TrainingRun.create(TEXT, TEXT, CONFIG)
    model = run.model.eval()
    a = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
    b = a.clone()
    b[:, 5:] = 12
    with torch.no_grad():
        la = model(a)
        lb = model(b)
        torch.testing.assert_close(la[:, :5], lb[:, :5], rtol=0, atol=0)
        loss, logits = model(a, targets=b)
        independent = (-torch.log_softmax(logits, dim=-1).gather(-1, b.unsqueeze(-1))).mean()
        torch.testing.assert_close(loss, independent)
    model.train()
    loss, _ = model(a, targets=b)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.embedding.token_embedding.weight.grad.abs().sum() > 0
