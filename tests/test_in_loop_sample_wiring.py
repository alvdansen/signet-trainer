"""OFFL-02 in-loop validation-sample wiring — pure helpers + train-seam text scan (D-9-OFFL02-CLOSE).

Three zero-GPU layers (mirroring ``test_offload_suspend_seam.py`` + ``test_multi_frame_sample_wiring.py``):

  * PURE HELPERS — ``in_loop_decoder_enabled`` / ``in_loop_sample_due`` are unit-tested against their
    truth tables (incl. the empty-prompts and off-cadence cases) with light ``SimpleNamespace`` stubs.
    They import WITHOUT pulling modal / ltx_* / GPU (a subprocess import-purity guard proves it).

  * TRAIN-SEAM TEXT SCAN — ``fns.py::train`` drives ``with_video_vae_decoder`` from the config knob,
    gates the in-loop sample via ``in_loop_sample_due``, pre-encodes prompts into
    ``CachedPromptEmbeddings`` (PHASE A, two-phase VRAM), and feeds ``cached_embeddings=`` to
    ``run_sampler`` — the OOM-prone raw ``run_sampler(components, model, vcfg, device=device)`` call
    is GONE from the in-loop body.

The LIVE GPU proof (in-loop sample under active block-swap) rides the gated campaign run (Wave 4); this
covers the CPU-verifiable wiring + decision surface ONLY. Comments + docstrings are stripped before
the text scan so prose mentions don't false-positive.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from signet_trainer.train.loop import in_loop_decoder_enabled, in_loop_sample_due

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FNS = _REPO_ROOT / "src" / "signet_trainer" / "modal" / "fns.py"


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove ``# ...`` comments + triple-quoted strings so prose doesn't trip the scan."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


# --------------------------------------------------------------------------------------------------
# Pure helper truth tables
# --------------------------------------------------------------------------------------------------


def test_in_loop_decoder_enabled_truth_table() -> None:
    """Decoder loads iff in_loop_sampling AND prompts non-empty (config-first, D-NOHARDCODE)."""
    assert in_loop_decoder_enabled(SimpleNamespace(in_loop_sampling=True, prompts=["a"])) is True
    # knob off -> decoder off even with prompts present.
    assert in_loop_decoder_enabled(SimpleNamespace(in_loop_sampling=False, prompts=["a"])) is False
    # empty prompts -> decoder off even with the knob set (defensive; schema also rejects this).
    assert in_loop_decoder_enabled(SimpleNamespace(in_loop_sampling=True, prompts=[])) is False
    assert in_loop_decoder_enabled(SimpleNamespace(in_loop_sampling=False, prompts=[])) is False


def test_in_loop_sample_due_truth_table() -> None:
    """Cadence gate: has_prompts AND decoder_ready AND step % checkpoint_every == 0."""
    # due — on cadence, decoder ready, prompt present.
    assert in_loop_sample_due(200, 200, True, True) is True
    assert in_loop_sample_due(400, 200, True, True) is True
    assert in_loop_sample_due(0, 200, True, True) is True  # step 0 is on cadence (0 % N == 0)
    # off cadence.
    assert in_loop_sample_due(150, 200, True, True) is False
    # decoder not ready (loader did not load it — in_loop_decoder_enabled was False).
    assert in_loop_sample_due(200, 200, False, True) is False
    # no prompt to render.
    assert in_loop_sample_due(200, 200, True, False) is False
    # defensive: checkpoint_every <= 0 never fires (no modulo-by-zero).
    assert in_loop_sample_due(200, 0, True, True) is False


def test_helpers_do_not_pull_modal_or_ltx() -> None:
    """Acceptance: `from signet_trainer.train.loop import ...` stays CPU-clean (no modal / no ltx_*)."""
    code = (
        "import sys; import signet_trainer.train.loop as m; "
        "assert callable(m.in_loop_decoder_enabled) and callable(m.in_loop_sample_due); "
        "assert 'modal' not in sys.modules, 'loop pulled modal'; "
        "assert not any(k == 'ltx_core' or k.startswith('ltx_') for k in sys.modules), "
        "'loop pulled ltx_*'; print('ok')"
    )
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# --------------------------------------------------------------------------------------------------
# Train-seam text scan (the OFFL-02 live wiring in fns.py::train)
# --------------------------------------------------------------------------------------------------


def test_train_seam_drives_decoder_from_config() -> None:
    """The train loader loads the decoder from the config knob (not the implicit default False)."""
    code = _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))
    # The decoder decision is bound from the config knob via the pure helper ...
    assert "in_loop_decoder_enabled(config.validation)" in code, (
        "train() must derive the in-loop decision from in_loop_decoder_enabled(config.validation)"
    )
    # ... and threaded into the loader (config-first, not the implicit default False).
    assert re.search(r"with_video_vae_decoder=\w+", code), (
        "train() must pass with_video_vae_decoder=<config-driven> to load_ltxv_components"
    )


def test_train_seam_gates_and_feeds_cached_embeddings() -> None:
    """The in-loop body gates via in_loop_sample_due and feeds cached embeddings (no Gemma re-encode)."""
    code = _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))
    assert "in_loop_sample_due(" in code, "the in-loop seam must gate via in_loop_sample_due"
    assert "cached_embeddings=cached_by_prompt[" in code, (
        "the in-loop sampler call must feed cached embeddings from the PHASE-A cache"
    )
    # PHASE-A pre-encode must be present (prompts encoded BEFORE Gemma is freed).
    assert "CachedPromptEmbeddings" in code, (
        "train() must pre-encode prompts into CachedPromptEmbeddings (PHASE A, two-phase VRAM)"
    )
    # The OOM-prone raw call (re-encodes via the freed Gemma) must be GONE from the in-loop body.
    assert "run_sampler(components, model, vcfg, device=device)" not in code, (
        "the raw run_sampler(components, model, vcfg, device=device) call must be removed "
        "(it would re-encode via the freed Gemma -> ~72GB OOM)"
    )


def test_offloader_suspend_wrapper_preserved() -> None:
    """The suspend/re-arm wrapper stays around the in-loop sample (active block-swap safety)."""
    code = _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))
    assert "with offloader_suspended(offloader, block_list):" in code, (
        "the in-loop sample must stay inside the offloader_suspended() wrapper (OFFL-02)"
    )


# --------------------------------------------------------------------------------------------------
# Mid-run cadence placement (the r1 finding, 2026-07-11 — D-9-OFFL02-CLOSE)
# --------------------------------------------------------------------------------------------------

_LOOP = _REPO_ROOT / "src" / "signet_trainer" / "train" / "loop.py"


def test_train_loop_exposes_on_checkpoint_callback() -> None:
    """train_loop must accept on_checkpoint and invoke it with the live step at cadence.

    The r1 run proved the pre-loop 'structural stand-in' rendered an UNTRAINED-adapter mp4
    named with max_steps — the mid-run cadence never fired. The callback is the fix: invoked
    AFTER ckpt_manager.save and BEFORE checkpoints_vol.commit() so the mp4 rides the same
    Volume commit as its checkpoint.
    """
    import inspect

    from signet_trainer.train.loop import train_loop

    assert "on_checkpoint" in inspect.signature(train_loop).parameters, (
        "train_loop must accept an on_checkpoint callback"
    )
    code = _strip_comments_and_docstrings(_LOOP.read_text(encoding="utf-8"))
    assert "on_checkpoint(global_step)" in code, (
        "train_loop must invoke on_checkpoint with the LIVE global_step at the cadence branch"
    )
    # Ordering (audit #12, 09.1-05): save -> commit -> callback -> commit. The checkpoint is
    # durable on the Volume BEFORE the ~8-13 min in-loop render; the mp4 rides a second commit.
    cadence = code[code.index("checkpoint_every == 0"):]
    i_save = cadence.index("ckpt_manager.save")
    i_cb = cadence.index("on_checkpoint(global_step)")
    i_commit1 = cadence.index("checkpoints_vol.commit()")
    i_commit2 = cadence.index("checkpoints_vol.commit()", i_cb)
    assert i_save < i_commit1 < i_cb < i_commit2, (
        "cadence branch must run ckpt save -> vol.commit -> on_checkpoint -> vol.commit "
        "(checkpoint durable before the render; mp4 rides the second commit — audit #12)"
    )


def test_train_wires_callback_not_preloop_standin() -> None:
    """fns.train must pass on_checkpoint=_in_loop_sample and carry NO pre-loop max_steps stand-in."""
    code = _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))
    assert "on_checkpoint=_in_loop_sample" in code, (
        "train() must wire the in-loop sampler into train_loop via on_checkpoint"
    )
    # The pre-loop structural stand-in (step = config.training.max_steps before the loop) is GONE.
    assert "step = config.training.max_steps" not in code, (
        "the pre-loop structural stand-in must be removed — samples fire mid-run with real steps"
    )
