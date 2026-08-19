"""LTX-2.5 Stage 1 (issue #53) entrypoint dispatch — behavioral, mirrors test_entrypoint_gate_
behavioral.py's shape exactly (WR-05): actually CALL main() and assert dispatch ROUTES to
``ltx25_train``/``ltx25_preprocess`` (never the plain 2.3 ``train``/``preprocess``) for a
``model.ltx_generation: '2.5'`` config, with zero real Modal spend (recording stubs).

Anti-pollution discipline (issue #45/#73, PR #78/#81): ``entrypoint.append_spend`` is neutralized
in every test that drives a real dispatch, and every config below carries EXPLICIT generous
``cost_guardrail_usd``/``session_cap_usd`` so the worst-case-ceiling pricing in ``ModalConfig``
never asks/refuses first — these tests exercise dispatch ROUTING, not the guardrail/ledger.

IMPORT DISCIPLINE (Anti-Pattern 6): modal-touching imports are done INSIDE the test bodies /
helpers, never at module top, so pytest COLLECTION never pulls ``modal`` into ``sys.modules``.
"""

from __future__ import annotations

import builtins

_LTX25_CONFIG = """
training_dims: [768, 352, 25]
model:
  ltx_generation: "2.5"
data:
  preprocessed_data_root: "precomputed"
training:
  max_steps: 10
modal:
  # explicit generous values (PR #78/#81 discipline): these tests exercise dispatch ROUTING, not
  # the guardrail/session-cap ceiling pricing.
  cost_guardrail_usd: 500.0
  session_cap_usd: 500.0
"""

_LTX23_CONFIG = """
training_dims: [768, 352, 25]
data:
  preprocessed_data_root: "precomputed"
training:
  max_steps: 10
modal:
  cost_guardrail_usd: 500.0
  session_cap_usd: 500.0
"""


class _RecordingCall:
    """Stand-in for a ``modal.FunctionCall`` (mirrors test_entrypoint_gate_behavioral.py)."""

    object_id = "fc-test-stub"

    def get(self, timeout: float | None = None) -> None:
        return None


class _RecordingFn:
    """Stand-in for a Modal Function — records ``.spawn``/``.with_options`` calls."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.with_options_calls: list[dict] = []

    def with_options(self, **kwargs) -> "_RecordingFn":
        self.with_options_calls.append(kwargs)
        return self

    def spawn(self, *args, **kwargs) -> _RecordingCall:
        self.calls.append((args, kwargs))
        return _RecordingCall()


def _raw_main():
    from signet_trainer.modal import entrypoint  # lazy — keep modal out of collection-time sys.modules

    return entrypoint, entrypoint.main.info.raw_f


def _write_config(tmp_path, text: str) -> str:
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(text, encoding="utf-8")
    return str(cfg_path)


def _neutralize_ledger(monkeypatch, entrypoint) -> list[tuple]:
    ledger_calls: list[tuple] = []
    monkeypatch.setattr(entrypoint, "append_spend", lambda *a, **k: ledger_calls.append((a, k)))
    return ledger_calls


def test_ltx_generation_25_train_mode_dispatches_ltx25_train_not_the_23_arm(
    tmp_path, monkeypatch
) -> None:
    from signet_trainer.modal import fns

    entrypoint, raw_main = _raw_main()
    monkeypatch.setattr(entrypoint, "run_dryrun", lambda cfg, mode=None: 0)  # skip the heavy gate
    monkeypatch.setattr(builtins, "input", lambda *a, **k: (_ for _ in ()).throw(EOFError()))
    _neutralize_ledger(monkeypatch, entrypoint)

    ltx25_rec = _RecordingFn()
    ltx23_rec = _RecordingFn()
    monkeypatch.setattr(fns, "ltx25_train", ltx25_rec)
    monkeypatch.setattr(fns, "train", ltx23_rec)

    raw_main(config=_write_config(tmp_path, _LTX25_CONFIG), approve=True, mode="train")

    assert len(ltx25_rec.calls) == 1, (
        "a model.ltx_generation=='2.5' config on --mode train must dispatch ltx25_train.spawn() "
        "exactly once"
    )
    assert ltx23_rec.calls == [], "the plain LTX-2.3 train() arm must never be reached for gen 2.5"


def test_ltx_generation_23_train_mode_still_dispatches_the_plain_train_arm(
    tmp_path, monkeypatch
) -> None:
    """Byte-identity regression: an ordinary (no ``model.ltx_generation`` key) config still routes
    to the UNCHANGED ``train()`` arm — the new ltx25 elif branch must never intercept it."""
    from signet_trainer.modal import fns

    entrypoint, raw_main = _raw_main()
    monkeypatch.setattr(entrypoint, "run_dryrun", lambda cfg, mode=None: 0)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: (_ for _ in ()).throw(EOFError()))
    _neutralize_ledger(monkeypatch, entrypoint)

    ltx25_rec = _RecordingFn()
    ltx23_rec = _RecordingFn()
    monkeypatch.setattr(fns, "ltx25_train", ltx25_rec)
    monkeypatch.setattr(fns, "train", ltx23_rec)

    raw_main(config=_write_config(tmp_path, _LTX23_CONFIG), approve=True, mode="train")

    assert len(ltx23_rec.calls) == 1
    assert ltx25_rec.calls == []


def test_ltx_generation_25_preprocess_mode_dispatches_ltx25_preprocess(
    tmp_path, monkeypatch
) -> None:
    from signet_trainer.modal import fns

    entrypoint, raw_main = _raw_main()
    monkeypatch.setattr(entrypoint, "run_dryrun", lambda cfg, mode=None: 0)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: (_ for _ in ()).throw(EOFError()))
    _neutralize_ledger(monkeypatch, entrypoint)

    ltx25_rec = _RecordingFn()
    ltx23_rec = _RecordingFn()
    monkeypatch.setattr(fns, "ltx25_preprocess", ltx25_rec)
    monkeypatch.setattr(fns, "preprocess", ltx23_rec)

    raw_main(config=_write_config(tmp_path, _LTX25_CONFIG), approve=True, mode="preprocess")

    assert len(ltx25_rec.calls) == 1, (
        "a model.ltx_generation=='2.5' config on --mode preprocess must dispatch "
        "ltx25_preprocess.spawn() exactly once"
    )
    assert ltx23_rec.calls == [], "the plain LTX-2.3 preprocess() arm must never be reached"

    # The threaded kwargs (LTX25_STAGE1_DESIGN.md §5 delta 2) must carry the config's model_id /
    # text_encoder_id / ltx25 split-path fields, not the fn's own standalone defaults.
    _, kwargs = ltx25_rec.calls[0]
    assert kwargs["model_id"] == "ltx-2.3-22b-dev.safetensors"  # the config's default model_id
    assert kwargs["gemma_root"] == "gemma-3-12b-it"  # the config's default text_encoder_id
    assert kwargs["video_vae_path"] is None
    assert kwargs["audio_vae_path"] is None


def test_ltx_generation_25_sample_mode_is_refused_pre_dispatch(tmp_path, monkeypatch) -> None:
    """--mode sample on a gen-2.5 config must be refused by the REAL mode_gate (not a stub) —
    run_dryrun is NOT monkeypatched here, so this exercises the actual pre-dispatch refusal path."""
    import pytest

    from signet_trainer.modal import fns

    entrypoint, raw_main = _raw_main()
    monkeypatch.setattr(builtins, "input", lambda *a, **k: (_ for _ in ()).throw(EOFError()))
    _neutralize_ledger(monkeypatch, entrypoint)

    ltx25_like_rec = _RecordingFn()
    monkeypatch.setattr(fns, "sample", ltx25_like_rec)

    with pytest.raises(SystemExit):
        raw_main(config=_write_config(tmp_path, _LTX25_CONFIG), approve=True, mode="sample")

    assert ltx25_like_rec.calls == [], "no dispatch may happen once the mode gate refuses"
