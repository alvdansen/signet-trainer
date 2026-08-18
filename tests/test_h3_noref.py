"""CPU contract tests for H3 NO-REFERENCE training (ALPHA — ``h3.references_per_sample: 0``).

The no-reference GEOMETRY was proven in Phase 10 (the 124f no-reference t2v baseline priced at
37,806 packed rows through the same budget math) and the primitives already guard the zero case
(``h3_row_timesteps``'s ``if n_cond_video:``, ``h3_loss_mask`` at ``n_cond_video=0``). What this
file pins is the SURFACE built on top of them:

  1. **The schema allowlist is {0, 2}** — 0 = no-reference (ALPHA), 2 = the Ref2VA operator ruling
     (PR #6 adds 1 for single-control; one-token union on merge). The exactly-2 refusal keeps its
     substitution-rule message.
  2. **The reverse guard**: a ref-only knob (reference_dropout, reference_pair_seed, ...) set
     alongside 0 dies at config load — with zero slots it acts on nothing, and a silently-ignored
     config block is the exact anti-pattern the lean field-split exists to prevent.
  3. **The strategy at zero slots**: the reference source is NEVER read (not declared, not looked
     up), the text conditions on the PROMPT-ONLY state, the packed batch carries zero
     conditioning-video rows through the SAME packer, and the loss mask covers every target row.
     A REF-BEARING cache stays consumable — its reference dir is simply never read.
  4. **The dry-run proves the reduced sequence**: zero ref rows, zero vision doubling (every image
     reference bills TWICE — vision tokens AND ref latent rows — so no-ref removes BOTH), and the
     budget totals over text + audio + target only. The example config passes end to end.
  5. **``--mode sample`` is REFUSED loudly** (entrypoint pre-approval + in-container): the
     transcribed diffusers workflow (ref2va) is reference-conditioned end to end, and the t2va
     workflow is not transcribed in this repo — a "ref2va with no references" render would be an
     unvalidated request under a no-reference label.

All tensors are CPU float32 and tiny. Nothing here imports ``diffusers`` or ``modal`` or touches
CUDA. All paths are SYNTHETIC (client-property hygiene).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from signet_trainer.config.schema import H3Config, SignetConfig
from signet_trainer.conditioning.h3_geometry import h3_packed_seq_len
from signet_trainer.conditioning.h3_ref import H3ModelInputs, H3RefStrategy
from signet_trainer.prep.h3_text_payload import build_h3_text_payload
from signet_trainer.train.h3_step import h3_step_deps_from_model

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FNS_SRC = _REPO_ROOT / "src" / "signet_trainer" / "modal" / "fns.py"
_ENTRYPOINT_SRC = _REPO_ROOT / "src" / "signet_trainer" / "modal" / "entrypoint.py"
_EXAMPLE_CONFIG = _REPO_ROOT / "configs" / "h3_t2va_noref.example.yaml"

#: The required ALPHA banner, verbatim (ASCII-safe runtime string).
ALPHA_BANNER = (
    "H3 NO-REFERENCE TRAINING IS ALPHA - smoke-tested only, no end-to-end run exists; file issues"
)

# Deliberately tiny feature widths — every contract below is about ROWS, not channels
# (the test_h3_ref_strategy.py convention).
PATCH_DIM = 6
AUDIO_IN_CHANNELS = 4
TEXT_DIM = 5

N_TEXT = 12
N_PROMPT_ONLY_TEXT = 5
N_TARGET_VIDEO = 9
N_TARGET_AUDIO = 3

T_VIDEO = 0.5
T_AUDIO = 0.4

#: Non-empty spans on the REF-BEARING state, so selecting the wrong state fails loudly below.
VISION_SPANS: tuple[tuple[int, int], ...] = ((1, 5), (6, 10))


# --------------------------------------------------------------------------------------------------
# helpers — the test_h3_ref_strategy.py fixture pattern, at zero slots
# --------------------------------------------------------------------------------------------------


def _deps() -> object:
    class _Cfg:
        in_channels = PATCH_DIM
        patch_size = (1, 1, 1)
        audio_in_channels = AUDIO_IN_CHANNELS
        text_dim = TEXT_DIM

    class _Model:
        config = _Cfg()

    return h3_step_deps_from_model(_Model())


def _monotone_position_ids(*, seq_len: int, **_: object) -> torch.Tensor:
    ids = torch.zeros(seq_len, 3, dtype=torch.float64)
    ids[:, 0] = torch.arange(seq_len, dtype=torch.float64)
    return ids


def _noref_strategy(**overrides: object) -> H3RefStrategy:
    kwargs: dict = {
        "references_per_sample": 0,
        "audio_in_loss": False,
        "deps": _deps(),
        "position_ids_fn": _monotone_position_ids,
    }
    kwargs.update(overrides)
    return H3RefStrategy(**kwargs)


def _text_payload(*, ref_bearing: bool) -> dict:
    """A version-2 text payload. ``ref_bearing=True`` mimics a Ref2VA cache (distinct states);
    ``False`` mimics a no-reference cache (one prompt-only state serving both keys)."""
    generator = torch.Generator().manual_seed(3)
    prompt_only = torch.randn(1, N_PROMPT_ONLY_TEXT, TEXT_DIM, generator=generator)
    if ref_bearing:
        return build_h3_text_payload(
            hidden_states=torch.randn(1, N_TEXT, TEXT_DIM, generator=generator),
            vision_spans=VISION_SPANS,
            prompt_only_hidden_states=prompt_only,
            has_references=True,
        )
    return build_h3_text_payload(
        hidden_states=prompt_only,
        vision_spans=(),
        prompt_only_hidden_states=prompt_only,
        has_references=False,
    )


def _noref_batch(*, ref_bearing_cache: bool = False, with_reference_source: bool = False) -> dict:
    """One dataloader sample WITHOUT the reference source (the no-reference cache layout), or —
    with ``with_reference_source=True`` — a ref-bearing cache whose reference dir was loaded anyway
    (must be ignored, never consumed)."""
    generator = torch.Generator().manual_seed(7)
    batch: dict = {
        "segment_index": 0,
        "step": 0,
        "h3_latent_conditions": {
            "latents": torch.randn(1, N_TARGET_VIDEO, PATCH_DIM, generator=generator)
        },
        "h3_text_conditions": _text_payload(ref_bearing=ref_bearing_cache),
        "h3_audio_latent_conditions": {
            "latents": torch.randn(1, N_TARGET_AUDIO, AUDIO_IN_CHANNELS, generator=generator)
        },
        "t_video": T_VIDEO,
        "t_audio": T_AUDIO,
    }
    if with_reference_source:
        # A deliberately MALFORMED reference payload: if the no-reference path ever reads it, the
        # parse raises — proving "never read" rather than "read and discarded".
        batch["h3_ref_latent_conditions"] = {"references": [{"malformed": True}]}
    return batch


def _noref_config_payload(**over) -> dict:
    """A minimal VALID no-reference H3 config (the test_h3_config_schema.py payload shape at 0)."""
    payload = {
        "training_dims": [1344, 768, 22],
        "data": {
            "preprocessed_data_root": "/data/h3_noref_preprocessed",
            "batch_size": 1,
            "resolution_buckets": ["1344x768x22"],
        },
        "model": {"family": "h3"},
        "training": {"max_steps": 100},
        "h3": {"references_per_sample": 0},
    }
    payload.update(over)
    return payload


# ==================================================================================================
# 1. Schema — the {0, 2} allowlist and the ALPHA marking
# ==================================================================================================


def test_schema_accepts_zero_references_per_sample() -> None:
    assert H3Config(references_per_sample=0).references_per_sample == 0


def test_schema_still_defaults_to_two_and_rejects_the_rest() -> None:
    """Default 2 unchanged; the union is now {0, 1, 2, 3} (3 added 2026-08-16 for
    explicit-manifest sequence tasks), 4/-1 still die on the substitution rule."""
    assert H3Config().references_per_sample == 2
    for bad in (4, -1):
        with pytest.raises(ValidationError) as exc:
            H3Config(references_per_sample=bad)
        msg = str(exc.value)
        assert "2" in msg
        assert "substitut" in msg.lower()


def test_the_refusal_names_the_full_union() -> None:
    """The refusal documents every legal count: 0 = no-reference (ALPHA), 1 = single-control
    (#6, merged), 2 = Ref2VA, 3 = explicit-manifest sequence tasks."""
    with pytest.raises(ValidationError) as exc:
        H3Config(references_per_sample=4)
    msg = str(exc.value)
    assert "0" in msg and "ALPHA" in msg
    assert "single-control" in msg
    assert "Ref2VA" in msg
    assert "explicit-manifest" in msg


def test_the_field_description_carries_the_alpha_marking() -> None:
    description = H3Config.model_fields["references_per_sample"].description
    assert "ALPHA" in description
    assert "NO-REFERENCE" in description.upper()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reference_image_short_edge", 1024),
        ("reference_dropout", 0.1),
        ("reference_pair_seed", 7),
        ("environment_ref_last", False),
        ("t_visual_cond", 0.5),
        ("character_reference_sizes", [(832, 1248, "A")]),
        ("environment_reference_sizes", [(1344, 768, "E1")]),
    ],
)
def test_ref_only_fields_are_rejected_at_zero_slots(field: str, value: object) -> None:
    """The reverse guard: a ref-only knob at 0 slots acts on nothing -> config-load error."""
    with pytest.raises(ValidationError) as exc:
        H3Config(references_per_sample=0, **{field: value})
    msg = str(exc.value)
    assert field in msg
    assert "NO-REFERENCE" in msg


def test_live_fields_stay_free_at_zero_slots() -> None:
    """The budget triple / prompt estimate / aspect / audio ruling all still act at 0."""
    h3 = H3Config(
        references_per_sample=0,
        prompt_tokens_estimate=128,
        target_aspect=(4, 3),
        audio_in_loss=False,
        gpu_usable_gib=79.25,
        resident_gib=62.97,
        mib_per_packed_row=1.21,
    )
    assert h3.prompt_tokens_estimate == 128


def test_a_full_no_reference_signet_config_loads_and_prices_the_reduced_sequence() -> None:
    cfg = SignetConfig.model_validate(_noref_config_payload())
    assert cfg.h3.references_per_sample == 0
    # The family default LoRA targets resolve exactly as on the Ref2VA path.
    assert isinstance(cfg.lora.target_modules, str)


def test_reference_subject_ids_is_rejected_at_zero_slots() -> None:
    """Cross-block lean field-split: the render slot selector has nothing to select at 0."""
    with pytest.raises(ValidationError, match="reference_subject_ids"):
        SignetConfig.model_validate(
            _noref_config_payload(validation={"reference_subject_ids": ["A", "B"]})
        )


def test_declared_reference_sizes_alongside_zero_slots_die_loud() -> None:
    """A no-ref config cannot silently carry a reference corpus declaration."""
    with pytest.raises(ValidationError, match="character_reference_sizes"):
        SignetConfig.model_validate(
            _noref_config_payload(
                h3={
                    "references_per_sample": 0,
                    "character_reference_sizes": [(832, 1248, "A")],
                }
            )
        )


# ==================================================================================================
# 2. The strategy at zero slots
# ==================================================================================================


def test_get_data_sources_drops_the_reference_source_at_zero_slots() -> None:
    assert list(_noref_strategy().get_data_sources()) == [
        "h3_latents",
        "h3_conditions",
        "h3_audio_latents",
    ]


def test_get_data_sources_is_unchanged_at_two_slots() -> None:
    assert list(H3RefStrategy(references_per_sample=2).get_data_sources()) == [
        "h3_latents",
        "h3_conditions",
        "h3_reference_latents",
        "h3_audio_latents",
    ]


def test_prepare_packs_zero_cond_rows_and_an_all_target_loss_mask() -> None:
    inputs = _noref_strategy().prepare_training_inputs(_noref_batch())
    assert isinstance(inputs, H3ModelInputs)
    packed = inputs.packed
    assert packed.n_cond_video == 0
    assert inputs.ref_seq_len == 0
    assert inputs.references == ()
    assert inputs.reference_dropped is False
    # Loss mask covers EVERY target video row — there is no reference prefix to exclude.
    assert bool(inputs.video_loss_mask.all())
    assert int(inputs.video_loss_mask.sum()) == N_TARGET_VIDEO
    # video_targets stays TARGET-ONLY (trivially the whole video span here).
    assert tuple(inputs.video_targets.shape) == (1, N_TARGET_VIDEO, PATCH_DIM)


def test_prepare_budget_math_on_the_reduced_sequence() -> None:
    """seq_len = prompt-only text + audio + target video — no ref rows, no vision doubling."""
    inputs = _noref_strategy().prepare_training_inputs(_noref_batch())
    packed = inputs.packed
    assert packed.n_text == N_PROMPT_ONLY_TEXT
    assert packed.seq_len == N_PROMPT_ONLY_TEXT + N_TARGET_AUDIO + N_TARGET_VIDEO
    # Zero vision rows: every row in the text span is tagged TEXT (no vision-token doubling —
    # a vision block inside the text stream would be tagged VIDEO).
    from signet_trainer.train.h3_step import H3_TEXT_TAG

    tags = packed.kwargs["token_tags"]
    text_indices = packed.kwargs["text_indices"]
    assert bool((tags[text_indices] == H3_TEXT_TAG).all())


def test_prepare_selects_the_prompt_only_state_from_a_ref_bearing_cache() -> None:
    """A REF-BEARING cache stays consumable: the no-ref train conditions on its PROMPT-ONLY state
    (no vision blocks describing references this run never shows)."""
    batch = _noref_batch(ref_bearing_cache=True)
    payload = batch["h3_text_conditions"]
    inputs = _noref_strategy().prepare_training_inputs(batch)
    expected = payload["prompt_only_hidden_states"]
    assert inputs.packed.n_text == expected.shape[1]
    assert torch.equal(inputs.video.context, expected)


def test_prepare_never_reads_the_reference_source() -> None:
    """A malformed reference payload in the batch must NOT raise — proof the source is never read
    (not merely read-and-discarded)."""
    inputs = _noref_strategy().prepare_training_inputs(
        _noref_batch(ref_bearing_cache=True, with_reference_source=True)
    )
    assert inputs.packed.n_cond_video == 0


def test_compute_loss_runs_without_a_reference_prefix() -> None:
    strategy = _noref_strategy()
    inputs = strategy.prepare_training_inputs(_noref_batch())
    video_out = torch.randn(1, N_TARGET_VIDEO, PATCH_DIM)
    audio_out = torch.randn(1, N_TARGET_AUDIO, AUDIO_IN_CHANNELS)
    loss = strategy.compute_loss(inputs, (video_out, audio_out))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_geometry_prices_zero_reference_layout() -> None:
    """The shared budget math at zero references: no vision rows, text = the bare prompt."""
    layout = h3_packed_seq_len(22, (16, 9), (), 96, 896)
    assert layout.n_cond_video == 0
    assert layout.n_vision == 0
    assert layout.n_text == 96
    assert layout.total == layout.n_text + layout.n_target_audio + layout.n_target_video


# ==================================================================================================
# 3. The dry-run gate on the example config
# ==================================================================================================


def test_the_example_config_exists_and_is_alpha_marked() -> None:
    text = _EXAMPLE_CONFIG.read_text(encoding="utf-8")
    assert "ALPHA" in text
    assert "references_per_sample: 0" in text


def test_the_example_config_loads_and_passes_the_dryrun(capsys: pytest.CaptureFixture) -> None:
    from signet_trainer.config.load import load_config
    from signet_trainer.dryrun.shapes import run_dryrun

    cfg = load_config(_EXAMPLE_CONFIG)
    assert cfg.model.family == "h3"
    assert cfg.h3.references_per_sample == 0
    assert run_dryrun(cfg) == 0
    banner = capsys.readouterr().out
    assert "NO-REFERENCE" in banner
    assert "ALPHA" in banner
    assert "ref video 0" in banner
    assert "incl 0 vision" in banner


def test_the_dryrun_batch_realizes_zero_cond_rows_on_the_example_config() -> None:
    from signet_trainer.config.load import load_config
    from signet_trainer.dryrun.shapes import build_h3_dryrun_batch

    from signet_trainer.train.h3_step import H3_TEXT_TAG

    batch = build_h3_dryrun_batch(load_config(_EXAMPLE_CONFIG))
    assert batch.n_cond_video == 0
    # No vision doubling: the whole text span is tagged TEXT.
    tags = batch.kwargs["token_tags"]
    assert bool((tags[batch.kwargs["text_indices"]] == H3_TEXT_TAG).all())
    assert bool(batch.video_loss_mask.all())


# ==================================================================================================
# 4. The refusing sample surface + the ALPHA runtime banner (source scans — no modal import)
# ==================================================================================================


def test_h3_sample_refuses_no_reference_rendering() -> None:
    source = _FNS_SRC.read_text(encoding="utf-8")
    start = source.index("def h3_sample(")
    body = source[start:]
    assert "references_per_sample == 0" in body
    assert "NOT supported" in body, "the no-reference render refusal must be loud and explicit"
    # The refusal fires at the config-revalidate step, BEFORE any component load.
    assert body.index("NOT supported") < body.index("from diffusers import ComponentsManager")


def test_the_entrypoint_refuses_no_reference_sample_pre_approval() -> None:
    source = _ENTRYPOINT_SRC.read_text(encoding="utf-8")
    assert "references_per_sample == 0" in source
    refusal = source.index("no-reference H3 rendering is NOT supported")
    approval = source.index("_require_approval(approve)")
    assert refusal < approval, "the sample refusal must fire BEFORE the approval gate (zero-spend)"


def test_the_alpha_banner_is_printed_at_the_front_of_a_no_ref_dispatch() -> None:
    """The exact ALPHA sentence appears in the entrypoint (pre-approval), h3_train and
    h3_preprocess (in-container) — ASCII-safe, verbatim."""
    for path in (_ENTRYPOINT_SRC, _FNS_SRC):
        source = path.read_text(encoding="utf-8")
        compact = re.sub(r"\s+", " ", source.replace('"', ""))
        assert ALPHA_BANNER in compact, f"{path.name} is missing the verbatim ALPHA banner"
    fns = _FNS_SRC.read_text(encoding="utf-8")
    for fn in ("def h3_train", "def h3_preprocess"):
        segment_start = fns.index(fn)
        segment = fns[segment_start : segment_start + 12000]
        assert "IS ALPHA" in segment, f"{fn} does not print the ALPHA banner at its front"


def test_no_new_entrypoint_mode_was_added() -> None:
    """Single-gate law: no-ref rides --mode train/sample/preprocess; the mode tuple is untouched."""
    source = _ENTRYPOINT_SRC.read_text(encoding="utf-8")
    assert (
        'if mode not in ("train", "sample", "preprocess", "fuse", "restore", "backup"):' in source
    )
