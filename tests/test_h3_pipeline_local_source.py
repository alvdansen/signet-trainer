"""D-10-DEF-14 — the H3 render loads every component from the VOLUME, never from the Hub.

The first-ever H3 render died in 73 s because ``h3_sample`` handed
``ModularPipeline.from_pretrained`` the value of ``model.model_id`` — ``minimax-h3/transformer_ref``,
the transformer PARTITION. That is the correct meaning of the field for ``h3_train`` and
``h3_loader`` and the wrong one for a pipeline ROOT, so the load could not find an index at all.

⛔ **The loud failure is not the dangerous one.** Re-pointing the field at the root clears the index
error and buys a MUCH worse outcome: this checkpoint's index records
``pretrained_model_name_or_path: MiniMaxAI/MiniMax-H3`` for every component, so the pipeline would
pull ~134 GiB from the Hub while standing on the Volume that already holds it — inside a metered
A100 container, with no exception anywhere and the bill as the only symptom. Every assertion here
exists because that failure is silent.

The index in ``H3_INDEX`` below is TRANSCRIBED from the real
``signe-trainer-weights:/minimax-h3/model_index.json``, read off the Volume — including the trap
that makes this whole module necessary: the file is NAMED ``model_index.json`` while carrying the
MODULAR three-element shape, which diffusers' ``model_index.json`` fallback silently skips.

CPU-only, zero network, zero spend.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from signet_trainer.inference.h3_pipeline_source import (
    H3_DECODE_FLOOR_FRAMES,
    H3_REF2VA_TRANSFORMER,
    assert_h3_components_loaded_locally,
    assert_h3_frame_count_is_renderable,
    assert_h3_sources_are_local,
    h3_aligned_num_frames,
    h3_decoder_num_chunks,
    h3_renderable_frame_bounds,
    h3_video_latent_num_frames,
    read_h3_pipeline_index,
    resolve_h3_component_sources,
)

REPO = Path(__file__).resolve().parents[1]
FNS = REPO / "src" / "signet_trainer" / "modal" / "fns.py"

#: The components the ``ref2va`` workflow loads, transcribed from the pinned diffusers blocks
#: (``minimax_h3/encoders.py``, ``decoders.py``, ``denoise.py``, ``before_denoise.py``).
#: ``image_processor`` / ``video_processor`` are ``from_config`` and are deliberately absent.
REF2VA_COMPONENTS = [
    "text_encoder",
    "tokenizer",
    "processor",
    "vae",
    "audio_vae",
    H3_REF2VA_TRANSFORMER,
    "scheduler",
    "audio_scheduler",
]

#: Transcribed from the Volume. Note every `pretrained_model_name_or_path`: the Hub id is what makes
#: the naive fix expensive rather than merely wrong.
H3_INDEX = {
    "_class_name": "MiniMaxH3ModularPipeline",
    "_diffusers_version": "0.36.0.dev0",
    "_blocks_class_name": "MiniMaxH3Blocks",
    **{
        name: [
            library,
            class_name,
            {
                "type_hint": [library, class_name],
                "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
                "subfolder": name,
                "variant": None,
                "revision": None,
            },
        ]
        for name, library, class_name in [
            ("text_encoder", "transformers", "Qwen3VLForConditionalGeneration"),
            ("tokenizer", "transformers", "Qwen2TokenizerFast"),
            ("processor", "transformers", "Qwen3VLProcessor"),
            ("vae", "diffusers", "AutoencoderKLMiniMaxH3"),
            ("audio_vae", "diffusers", "AutoencoderKLMiniMaxH3Audio"),
            # BOTH partitions are declared by the one index. Only `transformer_ref` is on our
            # Volume, and only `transformer_ref` is what ref2va denoises against.
            ("transformer", "diffusers", "MiniMaxH3Transformer3DModel"),
            ("transformer_ref", "diffusers", "MiniMaxH3Transformer3DModel"),
            ("scheduler", "diffusers", "MiniMaxH3Scheduler"),
            ("audio_scheduler", "diffusers", "MiniMaxH3Scheduler"),
        ]
    },
}

#: What the Volume actually holds under `/weights/minimax-h3` (`modal volume ls`, 2026-08-06).
VOLUME_PARTITIONS = [
    "transformer_ref",
    "scheduler",
    "processor",
    "audio_scheduler",
    "tokenizer",
    "audio_vae",
    "vae",
    "text_encoder",
]


@pytest.fixture
def h3_root(tmp_path: Path) -> Path:
    """A stand-in for the mounted `/weights/minimax-h3`, with the REAL partition set."""
    root = tmp_path / "minimax-h3"
    root.mkdir()
    (root / "model_index.json").write_text(json.dumps(H3_INDEX), encoding="utf-8")
    for name in VOLUME_PARTITIONS:
        (root / name).mkdir()
    return root


# ── the resolver ─────────────────────────────────────────────────────────────────────────────────


def test_every_ref2va_component_resolves_to_a_local_directory(h3_root: Path) -> None:
    index, path = read_h3_pipeline_index(h3_root)
    assert path.name == "model_index.json"
    sources = resolve_h3_component_sources(index, h3_root, REF2VA_COMPONENTS)
    assert set(sources) == set(REF2VA_COMPONENTS)
    for name, source in sources.items():
        assert source.local_dir == h3_root / name
        assert source.local_dir.is_dir()
        # The Hub id is CARRIED but never used as a location — it exists to be named in a refusal.
        assert source.declared_hub_id == "MiniMaxAI/MiniMax-H3"


def test_the_egress_guard_passes_only_when_every_component_is_on_the_volume(h3_root: Path) -> None:
    index, _ = read_h3_pipeline_index(h3_root)
    sources = resolve_h3_component_sources(index, h3_root, REF2VA_COMPONENTS)
    line = assert_h3_sources_are_local(sources, h3_root)
    assert "zero Hub egress" in line


def test_a_partition_missing_from_the_volume_is_refused_and_names_the_hub_id(h3_root: Path) -> None:
    """THE assertion. A missing partition is exactly when diffusers would reach for the Hub."""
    index, _ = read_h3_pipeline_index(h3_root)
    sources = resolve_h3_component_sources(index, h3_root, REF2VA_COMPONENTS)
    (h3_root / "vae").rmdir()
    with pytest.raises(RuntimeError) as excinfo:
        assert_h3_sources_are_local(sources, h3_root)
    message = str(excinfo.value)
    assert "vae" in message
    assert "MiniMaxAI/MiniMax-H3" in message
    assert "egress" in message


def test_the_transformer_partition_we_do_not_hold_is_refused(h3_root: Path) -> None:
    """`transformer` is declared by the index and is NOT on the Volume — the second reason the
    config-only fix fails. Asking for it must be loud, not a download."""
    index, _ = read_h3_pipeline_index(h3_root)
    sources = resolve_h3_component_sources(index, h3_root, ["transformer"])
    with pytest.raises(RuntimeError, match="transformer"):
        assert_h3_sources_are_local(sources, h3_root)


def test_an_empty_component_set_is_refused_rather_than_passing_vacuously(h3_root: Path) -> None:
    with pytest.raises(RuntimeError, match="vacuously|no components"):
        assert_h3_sources_are_local({}, h3_root)


def test_a_component_absent_from_the_index_is_refused(h3_root: Path) -> None:
    index, _ = read_h3_pipeline_index(h3_root)
    with pytest.raises(RuntimeError, match="declares no component"):
        resolve_h3_component_sources(index, h3_root, ["not_a_component"])


def test_a_two_element_plain_index_entry_is_refused(h3_root: Path) -> None:
    """The silent trap: a plain `model_index.json` entry is `(library, class_name)`, and diffusers'
    fallback turns it into a spec pointing at the ROOT — the wrong weights at a valid shape."""
    index, _ = read_h3_pipeline_index(h3_root)
    index["vae"] = ["diffusers", "AutoencoderKLMiniMaxH3"]
    with pytest.raises(RuntimeError, match="three-element"):
        resolve_h3_component_sources(index, h3_root, ["vae"])


def test_an_empty_subfolder_is_refused(h3_root: Path) -> None:
    index, _ = read_h3_pipeline_index(h3_root)
    index["vae"][2]["subfolder"] = ""
    with pytest.raises(RuntimeError, match="subfolder"):
        resolve_h3_component_sources(index, h3_root, ["vae"])


def test_a_subfolder_escaping_the_root_is_refused(h3_root: Path) -> None:
    index, _ = read_h3_pipeline_index(h3_root)
    index["vae"][2]["subfolder"] = "../elsewhere"
    with pytest.raises(RuntimeError, match="escapes"):
        resolve_h3_component_sources(index, h3_root, ["vae"])


def test_a_missing_index_names_the_root_it_looked_in(tmp_path: Path) -> None:
    root = tmp_path / "minimax-h3"
    root.mkdir()
    (root / "transformer_ref").mkdir()
    with pytest.raises(RuntimeError, match="modular_model_index.json"):
        read_h3_pipeline_index(root)


def test_the_partition_path_is_refused_as_a_root(h3_root: Path) -> None:
    """The literal D-10-DEF-14 defect: `model_id` (the partition) handed in as the root."""
    with pytest.raises(RuntimeError, match="model_index.json"):
        read_h3_pipeline_index(h3_root / "transformer_ref")


# ── the post-load half ───────────────────────────────────────────────────────────────────────────


def test_the_post_load_guard_accepts_the_local_root_and_refuses_a_hub_id(h3_root: Path) -> None:
    ok = {name: str(h3_root) for name in REF2VA_COMPONENTS}
    assert "no component resolved from the Hub" in assert_h3_components_loaded_locally(ok, h3_root)

    hubbed = dict(ok, vae="MiniMaxAI/MiniMax-H3")
    with pytest.raises(RuntimeError, match="MiniMaxAI/MiniMax-H3"):
        assert_h3_components_loaded_locally(hubbed, h3_root)


def test_a_component_that_never_loaded_is_refused(h3_root: Path) -> None:
    """`ModularPipeline.load_components` downgrades a failed load to a warning and leaves the
    attribute None. That is the shape this half catches."""
    partial = {name: str(h3_root) for name in REF2VA_COMPONENTS}
    partial["audio_vae"] = None
    with pytest.raises(RuntimeError, match="never loaded|NO source"):
        assert_h3_components_loaded_locally(partial, h3_root)


# ── what `h3_sample` actually does, on the AST ───────────────────────────────────────────────────


def _h3_sample_node() -> ast.FunctionDef:
    tree = ast.parse(FNS.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "h3_sample":
            return node
    raise AssertionError("h3_sample not found in modal/fns.py")


def _h3_sample_source() -> str:
    return ast.get_source_segment(FNS.read_text(encoding="utf-8"), _h3_sample_node()) or ""


def _attribute_chains() -> set[str]:
    """Every maximal ``a.b.c`` chain h3_sample actually EVALUATES.

    On the AST, never on the text: this function documents the defect it fixes in its own comments,
    so a text scan for `ModularPipeline.from_pretrained` matches the explanation of why that call is
    gone. Comment-stripping is not enough either — it does not reach an f-string literal.
    """
    chains: set[str] = set()

    def chain_of(node: ast.AST) -> str | None:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
            return ".".join(reversed(parts))
        return None

    for node in ast.walk(_h3_sample_node()):
        if isinstance(node, ast.Attribute):
            name = chain_of(node)
            if name:
                chains.add(name)
    return chains


def test_h3_sample_never_calls_modular_pipeline_from_pretrained() -> None:
    """The defect, pinned. `ModularPipeline.from_pretrained(<root>)` is the call whose per-component
    specs carry the Hub id — there is no argument that makes it safe here. And `load_components`
    downgrades a failed per-component load to a `logger.warning`, leaving the attribute None."""
    chains = _attribute_chains()
    assert "ModularPipeline.from_pretrained" not in chains
    assert not any(c.endswith(".load_components") for c in chains), sorted(chains)


def test_h3_sample_reads_pipeline_root_id_and_not_model_id_for_the_pipeline() -> None:
    chains = _attribute_chains()
    assert "config.model.pipeline_root_id" in chains
    # `model.model_id` means the transformer PARTITION to h3_train / h3_loader. h3_sample must not
    # read it at all — that overload is the defect.
    assert "config.model.model_id" not in chains


def test_h3_sample_drives_the_ref2va_transformer_partition() -> None:
    """`pipe.transformer` is a component the ref2va workflow never declares — it would be None."""
    source = _h3_sample_source()
    assert "H3_REF2VA_TRANSFORMER" in source
    assert "inject_lora(pipe.transformer," not in source
    assert "update_components(transformer=" not in source


def test_h3_sample_asserts_both_halves_of_the_egress_guard() -> None:
    source = _h3_sample_source()
    assert "assert_h3_sources_are_local" in source
    assert "assert_h3_components_loaded_locally" in source


def test_the_render_container_is_pinned_offline() -> None:
    """The structural half: `huggingface_hub` freezes the offline flag at import time, so it has to
    be an env var on the container rather than a line inside the function."""
    fns = FNS.read_text(encoding="utf-8")
    assert 'modal.Secret.from_dict({"HF_HUB_OFFLINE": "1"' in fns


# ── the configs ──────────────────────────────────────────────────────────────────────────────────


SAMPLE_CONFIGS = sorted((REPO / "configs").glob("h3_embe_r1_sample*.yaml"))


def test_the_sample_configs_exist() -> None:
    assert len(SAMPLE_CONFIGS) == 5


#: D-10-DEF-15 — which of the five eval configs MiniMax-H3 will actually generate, measured against
#: the pinned pipeline's own band rather than assumed. This is a RECORD of the blocker, not a
#: judgement about it: which lengths the eval should use is an operator decision.
#: (config frame_count -> renderable), from the first real dispatch (ap-oyrSEx8V1ydaGaNanj2fTI).
UNRENDERABLE_TODAY = {
    "h3_embe_r1_sample_short.yaml": 22,
    "h3_embe_r1_sample_ref_b029.yaml": 22,
    "h3_embe_r1_sample_ref_c018.yaml": 22,
    "h3_embe_r1_sample_mid.yaml": 56,
}


def test_the_render_band_is_the_pinned_one_and_not_the_training_frame_law() -> None:
    assert h3_renderable_frame_bounds() == (120, 360)
    # 22 is a legal `17n + 5` TRAINING bucket — this campaign's — and unrenderable. The two laws
    # are independent, which is the whole reason this defect reached a paid container.
    assert h3_aligned_num_frames(22) == 22
    with pytest.raises(RuntimeError, match="17n\\+5 training law|generation band|outside the"):
        assert_h3_frame_count_is_renderable(22, where="validation.frame_count")
    # 124 = 17*7 + 5 and 5.167 s — the only one of the five inside the band.
    assert "inside MiniMax-H3" in assert_h3_frame_count_is_renderable(
        124, where="validation.frame_count"
    )


def test_the_refusal_names_the_renderable_counts() -> None:
    with pytest.raises(RuntimeError) as excinfo:
        assert_h3_frame_count_is_renderable(56, where="validation.frame_count")
    message = str(excinfo.value)
    assert "124" in message  # the actionable list, not just a band
    assert "eval-design decision" in message


# ── The off-band waiver, and the decode floor it deliberately cannot reach ───────────────────────
#
# A campaign that trains STILLS stages them as the shortest legal `17n + 5` clip, so its trained
# length is off-band by construction. These pin the THREE laws apart: the 17n+5 form, the 5-15 s
# GENERATION BAND (policy — waivable), and the VIDEO VAE DECODE FLOOR (arithmetic — never waived).


def test_the_decode_floor_is_derived_from_the_vae_arithmetic_not_remembered() -> None:
    """22 is not a magic number: it is the shortest `17n + 5` the decoder yields a chunk for."""
    legal = [n for n in range(5, 200) if n % 17 == 5]
    decodable = [n for n in legal if h3_decoder_num_chunks(n) >= 1]
    assert min(decodable) == H3_DECODE_FLOOR_FRAMES == 22
    # Below it the failure is the EMPTY chunk list, not a small-but-valid decode.
    assert h3_decoder_num_chunks(5) == 0
    # The ENCODER is perfectly happy at 5 — which is why the round trip is asymmetric, and why
    # training at 5 is legal while rendering at 5 is not.
    assert h3_video_latent_num_frames(5) == 2
    assert h3_video_latent_num_frames(22) == 7


def test_allow_offband_waives_the_band_and_says_so_loudly() -> None:
    # 22 frames = 0.917 s: legal 17n+5, off-band, and above the decode floor.
    with pytest.raises(RuntimeError, match="outside the"):
        assert_h3_frame_count_is_renderable(22, where="validation.frame_count")
    allowed = assert_h3_frame_count_is_renderable(
        22, where="validation.frame_count", allow_offband=True
    )
    assert "OFF-BAND" in allowed
    # The waiver must not read as an endorsement: the checkpoint was RELEASED for 5-15 s, and an
    # off-band render's quality is the very thing being measured.
    assert "RELEASED" in allowed


def test_allow_offband_does_NOT_waive_the_decode_floor() -> None:
    """The point of the split: below 22 the failure is a crash, not a shorter clip."""
    for frames in (5, 22 - 17):  # 5 is the shortest legal training bucket
        with pytest.raises(RuntimeError, match="DECODE FLOOR") as excinfo:
            assert_h3_frame_count_is_renderable(
                frames, where="validation.frame_count", allow_offband=True
            )
        message = str(excinfo.value)
        # It must name WHERE it would have failed: the cost of learning this late is a fully
        # paid denoise loop.
        assert "torch.cat" in message
        assert "decode shim" in message


def test_a_render_inside_the_band_is_unchanged_by_the_flag() -> None:
    """The flag WIDENS, never replaces — so it cannot change a legal render's meaning."""
    assert assert_h3_frame_count_is_renderable(
        124, where="validation.frame_count"
    ) == assert_h3_frame_count_is_renderable(
        124, where="validation.frame_count", allow_offband=True
    )


@pytest.mark.parametrize("path", SAMPLE_CONFIGS, ids=lambda p: p.name)
def test_which_sample_configs_h3_will_generate(path: Path) -> None:
    """Pins the CURRENT state of D-10-DEF-15 so a length change is a deliberate edit here too."""
    pytest.importorskip("torch")
    from signet_trainer.config.load import load_config

    frames = int(load_config(path).validation.frame_count)
    if path.name in UNRENDERABLE_TODAY:
        assert frames == UNRENDERABLE_TODAY[path.name]
        with pytest.raises(RuntimeError, match="outside the"):
            assert_h3_frame_count_is_renderable(frames, where="validation.frame_count")
    else:
        assert_h3_frame_count_is_renderable(frames, where="validation.frame_count")


@pytest.mark.parametrize("path", SAMPLE_CONFIGS, ids=lambda p: p.name)
def test_every_sample_config_declares_the_pipeline_root_and_keeps_model_id(path: Path) -> None:
    pytest.importorskip("torch")
    from signet_trainer.config.load import load_config

    config = load_config(path)
    assert config.model.pipeline_root_id == "minimax-h3"
    # ⛔ UNCHANGED, and it must stay that way: h3_train and h3_loader read this field as the
    # transformer partition and are proven across five dispatches.
    assert config.model.model_id == "minimax-h3/transformer_ref"
    assert config.model.pipeline_root_id != config.model.model_id
