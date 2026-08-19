"""Phase 10 (gap closure) — D-10-DEF-2: the reference payload must carry its DESCRIPTORS.

``prep/h3_encode.encode_h3_reference_latents`` wrote sizes but no identity; ``conditioning/h3_ref.
_parse_reference_pool`` requires ``path`` / ``kind`` / ``subject_id`` on every slot. The seam failed
LOUD, which was correct, but it failed only after a whole metered pre-encode had written a cache the
trainer could not read — so the honest close is to propagate the metadata the manifest already
carries rather than to infer any of it.

**Nothing here guesses ``kind``.** ``kind`` decides D-10-REFORDER slot ordering, and because an
image reference consumes one integer unit of the shared rotary clock, a reordered reference set is a
genuinely different request — a wrong kind trains against conditioning nobody asked for, silently
and at a perfectly valid shape. It is READ from the manifest KEY (``character_references`` /
``environment_reference``) or declared per-entry; ``subject_id`` is read from the entry, whose
vocabulary the config already declares.

The round trip is exercised END TO END on CPU with a stub VAE — encode -> torch.save ->
``PrecomputedDataset`` -> ``_parse_reference_pool`` -> ``order_reference_slots`` — because every
individual contract can be right while the composition still drops a field. Zero GPU, zero Modal
spend, no diffusers.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from signet_trainer.conditioning.h3_geometry import H3_CANVAS_MULTIPLE  # noqa: E402
from signet_trainer.conditioning.h3_ref import (  # noqa: E402
    H3_REFERENCE_KINDS,
    _parse_reference_pool,
    order_reference_slots,
)
from signet_trainer.models.h3_loader import (  # noqa: E402
    EXPECTED_H3_IN_CHANNELS,
    EXPECTED_H3_PATCH_SIZE,
)
from signet_trainer.prep.h3_encode import (  # noqa: E402
    H3_REFERENCE_LATENTS_DIR,
    encode_h3_reference_latents,
    write_h3_precomputed,
)
from signet_trainer.prep.h3_vae_contract import H3VideoVaeContractStub  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
_FNS = REPO / "src" / "signet_trainer" / "modal" / "fns.py"

#: The VAE's spatial compression, DERIVED the way ``h3_latent_grid_of_reference`` derives it:
#: ``H3_CANVAS_MULTIPLE`` is ``vae_spatial_compression_ratio * patch_w``, so the VAE's own factor is
#: 32 // 2 = 16 and the extra 2 comes from the patchify. Restating 32 here would make the stub
#: produce odd-width latent grids the real encode never emits — a stub that cannot be patchified.
_VAE_SPATIAL_FACTOR = H3_CANVAS_MULTIPLE // EXPECTED_H3_PATCH_SIZE[2]


# ==================================================================================================
# CPU stubs — no diffusers, no Pillow, no GPU
#
# ⛔ The video VAE is NOT stubbed here any more. This file used to carry its own ``_StubVideoVae``
# whose ``encode`` unpacked ``_, frames, height, width = pixels.shape`` — i.e. it REQUIRED the 4-D
# input the real ``AutoencoderKLMiniMaxH3`` refuses, so it was green for the whole of D-10-DEF-9.
# The one sanctioned stub lives in ``prep/h3_vae_contract`` and is diffed probe-for-probe against
# the real class by ``tests/test_h3_vae_input_contract.py``.
# ==================================================================================================


class _StubImage:
    """The three things ``_reference_pixels`` uses: ``.size``, ``.convert``, ``.resize``."""

    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)

    def convert(self, _mode: str) -> _StubImage:
        return self

    def resize(self, size: tuple[int, int], _resample: object = None) -> _StubImage:
        return _StubImage(*size)

    def __array__(self, dtype: object = None, copy: object = None) -> object:  # noqa: ARG002
        width, height = self.size
        array = np.full((height, width, 3), 128, dtype=np.uint8)
        return array if dtype is None else array.astype(dtype)


_LATENT_STATS = (torch.zeros(EXPECTED_H3_IN_CHANNELS), torch.ones(EXPECTED_H3_IN_CHANNELS))

#: Two slots at DIFFERENT source sizes — the fact that forced a per-slot list in the first place.
#: Sizes only; subject ids are labels from the config's declared vocabulary, never filenames.
_CHARACTER = {"path": "refs/char_a.png", "kind": "character", "subject_id": "A"}
_ENVIRONMENT = {"path": "refs/env_029.png", "kind": "environment", "subject_id": "029"}


def _encode(
    descriptors: list[dict],
    sizes: list[tuple[int, int]] | None = None,
    *,
    explicit_manifest: bool = False,
) -> dict:
    sizes = sizes or [(1024, 1536), (1344, 768)]
    images = [_StubImage(w, h) for w, h in sizes[: len(descriptors)]]
    return encode_h3_reference_latents(
        H3VideoVaeContractStub(),
        images,
        896,
        *_LATENT_STATS,
        descriptors=descriptors,
        references_per_sample=len(descriptors),
        explicit_manifest=explicit_manifest,
    )


# ==================================================================================================
# The writer now emits what the reader requires
# ==================================================================================================


def test_every_encoded_slot_carries_path_kind_and_subject_id() -> None:
    """The three fields ``_parse_reference_pool`` requires, on every slot, from the encode."""
    payload = _encode([_CHARACTER, _ENVIRONMENT])
    slots = payload["references"]
    assert len(slots) == 2
    for slot, descriptor in zip(slots, (_CHARACTER, _ENVIRONMENT), strict=True):
        for field in ("path", "kind", "subject_id"):
            assert slot[field] == descriptor[field], f"slot lost its {field}"


def test_the_descriptor_is_paired_with_its_own_latents_not_another_slots() -> None:
    """Positional pairing, proven by the SIZES: slot 0 is the 1024x1536 character, slot 1 is not."""
    payload = _encode([_CHARACTER, _ENVIRONMENT], sizes=[(1024, 1536), (1344, 768)])
    slots = payload["references"]
    assert slots[0]["subject_id"] == "A"
    assert tuple(slots[0]["source_wh"]) == (1024, 1536)
    assert slots[1]["subject_id"] == "029"
    assert tuple(slots[1]["source_wh"]) == (1344, 768)


# ==================================================================================================
# issue #52 — the provenance flag: explicit-manifest vs pool-derived
# ==================================================================================================


def test_the_payload_defaults_to_pool_derived_when_the_caller_says_nothing() -> None:
    """A caller that never mentions ``explicit_manifest`` gets the pool-derived (``False``) shape —
    every OTHER caller of this function (the both-modalities smoke, hand-built fixtures) keeps
    writing exactly what it always has.
    """
    payload = _encode([_CHARACTER, _ENVIRONMENT])
    assert payload["explicit_manifest"] is False


def test_the_explicit_manifest_flag_is_recorded_verbatim() -> None:
    """The flag travels into the payload unchanged — the ONLY place it is decided is the caller."""
    payload = _encode([_CHARACTER, _ENVIRONMENT], explicit_manifest=True)
    assert payload["explicit_manifest"] is True

    payload = _encode([_CHARACTER, _ENVIRONMENT], explicit_manifest=False)
    assert payload["explicit_manifest"] is False


def test_the_flag_survives_a_save_and_reload_round_trip(tmp_path: Path) -> None:
    """``H3RefStrategy`` reads the flag off the SAME dict ``PrecomputedDataset`` hands it — prove
    the round trip a real training run takes, not just the in-memory dict this module builds.
    """
    from signet_trainer.data.precomputed import PrecomputedDataset

    payload = _encode([_CHARACTER, _ENVIRONMENT], explicit_manifest=True)
    write_h3_precomputed(tmp_path, Path("segment_000.pt"), references=payload)
    dataset = PrecomputedDataset(
        tmp_path, data_sources={H3_REFERENCE_LATENTS_DIR: "h3_ref_latent_conditions"}
    )
    assert dataset[0]["h3_ref_latent_conditions"]["explicit_manifest"] is True


def test_the_latent_grid_dims_are_not_the_source_pixel_size() -> None:
    """``height``/``width`` are the LATENT grid; ``source_wh`` is the source. They must differ here.

    If they were ever equal this test's whole point would evaporate — the SOURCE-vs-LATENT
    distinction is what makes the parser's ``source_wh`` preference load-bearing.
    """
    slot = _encode([_CHARACTER])["references"][0]
    assert (slot["width"], slot["height"]) != tuple(slot["source_wh"])
    assert slot["width"] < slot["source_wh"][0]


@pytest.mark.parametrize(
    ("descriptors", "match"),
    [
        pytest.param([_CHARACTER, {**_ENVIRONMENT, "kind": ""}], "missing", id="no-kind"),
        pytest.param(
            [_CHARACTER, {"path": "refs/e.png", "kind": "environment"}], "missing", id="no-subject"
        ),
        pytest.param(
            [_CHARACTER, {**_ENVIRONMENT, "kind": "scene"}], "expected one of", id="unknown-kind"
        ),
        pytest.param(
            [_CHARACTER, {**_ENVIRONMENT, "path": "/dataset/refs/e.png"}],
            "ABSOLUTE",
            id="absolute-path",
        ),
        pytest.param([_CHARACTER, dict(_CHARACTER)], "share path", id="duplicate-path"),
    ],
)
def test_the_encoder_refuses_a_descriptor_it_cannot_trust(
    descriptors: list[dict], match: str
) -> None:
    """Refusals happen BEFORE the VAE runs, so each costs nothing and names its own fix."""
    with pytest.raises((ValueError, TypeError), match=match):
        _encode(descriptors)


def test_a_bare_tensor_slot_is_still_refused_by_the_reader() -> None:
    """The reader's guard is not weakened by the writer being fixed — both ends stay strict."""
    with pytest.raises(ValueError, match="missing"):
        _parse_reference_pool({"references": [{"latents": torch.zeros(1, 4, 8)}]})


def test_the_reader_names_the_re_encode_when_a_descriptor_less_cache_is_read() -> None:
    """A cache written before this fix is UNREADABLE, and the refusal must say so.

    Nothing can repair such a cache in place: ``kind`` is not in it, is not inferable, and a guess
    reorders the references. The only correct instruction is 're-encode'.
    """
    with pytest.raises(ValueError, match="RE-ENCODED"):
        _parse_reference_pool(
            {"references": [{"latents": torch.zeros(1, 4, 8), "width": 8, "height": 8}]}
        )


# ==================================================================================================
# The full CPU round trip — encode -> disk -> PrecomputedDataset -> pool -> D-10-REFORDER
# ==================================================================================================


def test_the_descriptors_survive_the_precomputed_dataset_round_trip(tmp_path: Path) -> None:
    """Every contract can be individually right while the composition still drops a field.

    ``torch.save`` -> ``torch.load(weights_only=True)`` is the step that would silently drop a
    non-tensor payload field, and ``PrecomputedDataset`` routes this source through
    ``_normalize_video_latents``. Proving the strings arrive is the whole point.
    """
    from signet_trainer.data.precomputed import PrecomputedDataset

    payload = _encode([_CHARACTER, _ENVIRONMENT])
    write_h3_precomputed(tmp_path, Path("clips/segment_000.pt"), references=payload)

    dataset = PrecomputedDataset(
        tmp_path, data_sources={H3_REFERENCE_LATENTS_DIR: "h3_ref_latent_conditions"}
    )
    assert len(dataset) == 1
    loaded = dataset[0]["h3_ref_latent_conditions"]

    pool = _parse_reference_pool(loaded, EXPECTED_H3_PATCH_SIZE)
    assert [reference.subject_id for reference, _ in pool] == ["A", "029"]
    assert [reference.kind for reference, _ in pool] == ["character", "environment"]
    assert [reference.path for reference, _ in pool] == [_CHARACTER["path"], _ENVIRONMENT["path"]]


def test_the_round_tripped_reference_carries_SOURCE_pixels_not_the_latent_grid(
    tmp_path: Path,
) -> None:
    """``H3Reference.width``/``height`` mean SOURCE pixels — ``to_geometry_reference`` prices on them.

    Reading the payload's ``width``/``height`` (the latent grid) instead would type-check, order
    correctly, and under-price the reference by ~1000x the moment anything priced it.
    """
    from signet_trainer.data.precomputed import PrecomputedDataset

    payload = _encode([_CHARACTER, _ENVIRONMENT], sizes=[(1024, 1536), (1344, 768)])
    write_h3_precomputed(tmp_path, Path("segment_000.pt"), references=payload)
    dataset = PrecomputedDataset(
        tmp_path, data_sources={H3_REFERENCE_LATENTS_DIR: "h3_ref_latent_conditions"}
    )
    pool = _parse_reference_pool(
        dataset[0]["h3_ref_latent_conditions"], EXPECTED_H3_PATCH_SIZE
    )

    character, environment = (reference for reference, _ in pool)
    assert (character.width, character.height) == (1024, 1536)
    assert (environment.width, environment.height) == (1344, 768)
    assert character.to_geometry_reference().label == "A", (
        "the pricing label is the subject_id — a budget refusal names it (C+008)"
    )


def test_the_round_tripped_pool_orders_the_environment_last(tmp_path: Path) -> None:
    """D-10-REFORDER, end to end: the kind that came out of the MANIFEST is the kind that orders.

    Written deliberately environment-FIRST so the ordering has to actually move it. If ``kind`` had
    been guessed (or defaulted to "character"), this passes with the references in the wrong slots —
    which is precisely why it was not guessed.
    """
    from signet_trainer.data.precomputed import PrecomputedDataset

    payload = _encode([_ENVIRONMENT, _CHARACTER], sizes=[(1344, 768), (1024, 1536)])
    write_h3_precomputed(tmp_path, Path("segment_000.pt"), references=payload)
    dataset = PrecomputedDataset(
        tmp_path, data_sources={H3_REFERENCE_LATENTS_DIR: "h3_ref_latent_conditions"}
    )
    pool = _parse_reference_pool(
        dataset[0]["h3_ref_latent_conditions"], EXPECTED_H3_PATCH_SIZE
    )

    ordered = order_reference_slots([reference for reference, _ in pool])
    assert [r.kind for r in ordered] == ["character", "environment"]
    assert [r.subject_id for r in ordered] == ["A", "029"]


def test_the_descriptor_sizes_reproduce_the_rows_the_encode_actually_wrote(tmp_path: Path) -> None:
    """The strongest statement that ``source_wh`` is the right field to read.

    ``H3PositionIdsBuilder`` re-derives every reference's latent grid from the ``H3Reference``'s
    width/height and REFUSES a batch whose derivation disagrees with the rows measured off the real
    tensors. Feeding it the payload's latent-grid dims instead of ``source_wh`` would be off by the
    VAE's spatial factor and abort the first batch — after the pre-encode was paid for. This runs
    that exact derivation on CPU against the tensors the encode actually produced.
    """
    from signet_trainer.conditioning.h3_packing import h3_latent_grid_of_reference
    from signet_trainer.data.precomputed import PrecomputedDataset

    payload = _encode([_CHARACTER, _ENVIRONMENT])
    written = {slot["subject_id"]: int(slot["latent_rows"]) for slot in payload["references"]}
    write_h3_precomputed(tmp_path, Path("segment_000.pt"), references=payload)
    dataset = PrecomputedDataset(
        tmp_path, data_sources={H3_REFERENCE_LATENTS_DIR: "h3_ref_latent_conditions"}
    )
    pool = _parse_reference_pool(
        dataset[0]["h3_ref_latent_conditions"], EXPECTED_H3_PATCH_SIZE
    )

    _, patch_h, patch_w = EXPECTED_H3_PATCH_SIZE
    for reference, rows in pool:
        frames, height, width = h3_latent_grid_of_reference(
            reference.width, reference.height, 896, EXPECTED_H3_PATCH_SIZE
        )
        derived = frames * (height // patch_h) * (width // patch_w)
        assert derived == int(rows.shape[1]) == written[reference.subject_id], (
            f"reference {reference.subject_id} re-derives to {derived} row(s) but the encode wrote "
            f"{int(rows.shape[1])} — the descriptor size and the encoded tensor disagree"
        )


def test_the_slot_paths_are_unique_so_the_gather_cannot_collapse(tmp_path: Path) -> None:
    """``H3RefStrategy`` gathers reference rows through ``{reference.path: rows}``.

    Two slots sharing a path would collapse into one reference repeated twice — the same
    conditioning, at the right shape and the right row count, with nothing to notice it.
    """
    from signet_trainer.data.precomputed import PrecomputedDataset

    payload = _encode([_CHARACTER, _ENVIRONMENT])
    write_h3_precomputed(tmp_path, Path("segment_000.pt"), references=payload)
    dataset = PrecomputedDataset(
        tmp_path, data_sources={H3_REFERENCE_LATENTS_DIR: "h3_ref_latent_conditions"}
    )
    pool = _parse_reference_pool(
        dataset[0]["h3_ref_latent_conditions"], EXPECTED_H3_PATCH_SIZE
    )

    by_path = {reference.path: rows for reference, rows in pool}
    assert len(by_path) == len(pool), "a collapsed join key silently duplicates one reference"


# ==================================================================================================
# The manifest side — read, never inferred
# ==================================================================================================


def _resolver():
    """Import ``_h3_resolve_references`` WITHOUT importing ``modal/fns.py``.

    Importing that module builds the Modal app graph and eagerly resolves every ``Secret.from_name``.
    The two helpers under test are pure stdlib, so they are exec'd out of the parsed source instead.
    """
    tree = ast.parse(_FNS.read_text(encoding="utf-8"))
    wanted = {"_h3_reference_entry", "_h3_resolve_references"}
    module = ast.Module(
        body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted],
        type_ignores=[],
    )
    assert len(module.body) == len(wanted), f"expected {wanted} as top-level defs in fns.py"
    namespace: dict = {"Any": object}
    exec(compile(ast.fix_missing_locations(module), "<fns-slice>", "exec"), namespace)  # noqa: S102
    return namespace["_h3_resolve_references"]


def _row_explicit_manifest_fn():
    """Import ``_h3_row_is_explicit_manifest`` the same exec-a-slice way, for the same reason."""
    tree = ast.parse(_FNS.read_text(encoding="utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_h3_row_is_explicit_manifest"
    )
    module = ast.Module(body=[node], type_ignores=[])
    namespace: dict = {}
    exec(compile(ast.fix_missing_locations(module), "<fns-slice>", "exec"), namespace)  # noqa: S102
    return namespace["_h3_row_is_explicit_manifest"]


_POOL_ROW = {
    "character_references": [
        {"path": "refs/char_a.png", "subject_id": "A"},
        {"path": "refs/char_b.png", "subject_id": "B"},
    ],
    "environment_reference": {"path": "refs/env_029.png", "subject_id": "029"},
}

_EXPLICIT_ROW = {
    "reference_paths": [
        {"path": "refs/a.png", "subject_id": "A", "kind": "character"},
        {"path": "refs/b.png", "subject_id": "B", "kind": "character"},
    ]
}


def test_a_pool_row_is_not_explicit_manifest() -> None:
    """``character_references`` is the rotating pool — never positively marked explicit."""
    assert _row_explicit_manifest_fn()(_POOL_ROW) is False


def test_a_reference_paths_row_is_explicit_manifest() -> None:
    """``reference_paths`` is the exact, order-load-bearing set — issue #52's 'explicit' case."""
    assert _row_explicit_manifest_fn()(_EXPLICIT_ROW) is True


def test_an_empty_reference_paths_list_is_not_explicit_manifest() -> None:
    """An empty list is falsy — this row has no explicit references at all, so it is not the
    explicit-manifest case (and ``_h3_resolve_references`` falls through to the pool branch,
    which will refuse it for a different, unrelated reason)."""
    assert _row_explicit_manifest_fn()({"reference_paths": []}) is False


def test_kind_comes_from_the_manifest_key_not_from_a_guess() -> None:
    """``character_references`` / ``environment_reference`` SAY what their members are."""
    resolved = _resolver()(
        _POOL_ROW,
        0,
        Path("/dataset/h3_embe"),
        references_per_sample=2,
        reference_pair_seed=42,
        environment_ref_last=True,
    )
    assert [r["kind"] for r in resolved] == ["character", "environment"]
    assert resolved[-1]["subject_id"] == "029"
    assert all(r["kind"] in H3_REFERENCE_KINDS for r in resolved)


def test_the_cached_path_is_manifest_relative_and_the_source_stays_local() -> None:
    """``path`` travels into a committed payload; the resolved mount path must not."""
    resolved = _resolver()(
        _POOL_ROW,
        0,
        Path("/dataset/h3_embe"),
        references_per_sample=2,
        reference_pair_seed=42,
        environment_ref_last=True,
    )
    for slot in resolved:
        assert not Path(slot["path"]).is_absolute(), "a mount prefix must never reach the cache"
        assert str(slot["source"]).endswith(Path(slot["path"]).name)


def test_a_bare_path_string_entry_is_refused_with_the_shape_to_write() -> None:
    """The refusal must be actionable — an operator has to know what to put in the manifest."""
    row = {"character_references": ["refs/char_a.png", "refs/char_b.png"]}
    with pytest.raises(ValueError, match="subject_id"):
        _resolver()(
            row,
            0,
            Path("/dataset/h3_embe"),
            references_per_sample=2,
            reference_pair_seed=42,
            environment_ref_last=True,
        )


def test_a_pool_row_is_refused_at_three_slots() -> None:
    """MAJOR-1 (house audit, PR #51): 3+ slots are EXPLICIT-MANIFEST ONLY.

    The config-level widening to ``references_per_sample=3`` is safe only because the pool /
    round-robin branch is UNREACHABLE at 3 — every row must instead supply its own
    ``reference_paths``. Before this fix that safety case was prose: a ``character_references`` pool
    of exactly 3 sailed through and made conditioning CONSTANT across the corpus (the exact
    copy-collapse regime the 2-slot cap existed to prevent), and a pool of 2 silently duplicated a
    reference via the round-robin's modulo wraparound (``[c0, c1, c0]``) while the row-count check
    passed. Refusing the pool branch outright at ``references_per_sample >= 3`` is what makes the
    PR's safety case code, not prose.
    """
    row = {
        "character_references": [
            {"path": "refs/char_a.png", "subject_id": "A"},
            {"path": "refs/char_b.png", "subject_id": "B"},
            {"path": "refs/char_c.png", "subject_id": "C"},
        ]
    }
    with pytest.raises(ValueError, match="EXPLICIT-MANIFEST"):
        _resolver()(
            row,
            0,
            Path("/dataset/h3_embe"),
            references_per_sample=3,
            reference_pair_seed=42,
            environment_ref_last=True,
        )


def test_a_short_pool_is_refused_rather_than_silently_wrapping() -> None:
    """A pool smaller than the character slots it must fill would otherwise wrap and DUPLICATE a
    reference within the same sample — refused instead of padded (house audit, PR #51, MAJOR-1).

    Two-character pool, ``references_per_sample=2``, no environment reference: 2 character slots
    are needed but the pool holds only 1, so ``pool[(start + offset) % len(pool)]`` for
    ``offset in (0, 1)`` would pick the SAME entry twice.
    """
    row = {"character_references": [{"path": "refs/char_a.png", "subject_id": "A"}]}
    with pytest.raises(ValueError, match="modulo wraparound"):
        _resolver()(
            row,
            0,
            Path("/dataset/h3_embe"),
            references_per_sample=2,
            reference_pair_seed=42,
            environment_ref_last=True,
        )


def test_a_flat_reference_paths_list_must_declare_its_own_kinds() -> None:
    """``reference_paths`` carries no key to read a kind from, so the entries must say."""
    resolve = _resolver()
    with pytest.raises(ValueError, match="missing"):
        resolve(
            {"reference_paths": [{"path": "refs/a.png", "subject_id": "A"}]},
            0,
            Path("/dataset/h3_embe"),
            references_per_sample=1,
            reference_pair_seed=42,
            environment_ref_last=True,
        )
    ok = resolve(
        {"reference_paths": [{"path": "refs/a.png", "subject_id": "A", "kind": "prop"}]},
        0,
        Path("/dataset/h3_embe"),
        references_per_sample=1,
        reference_pair_seed=42,
        environment_ref_last=True,
    )
    assert ok[0]["kind"] == "prop"


# ==================================================================================================
# Wiring — the stage must actually thread the descriptors it resolved
# ==================================================================================================


def _stage_source(name: str) -> str:
    code = _FNS.read_text(encoding="utf-8")
    match = re.search(rf"^def {re.escape(name)}\(", code, re.M)
    assert match, f"{name}() not found in modal/fns.py"
    tail = re.search(r"^(?:def |class |@)", code[match.end() :], re.M)
    end = match.end() + tail.start() if tail else len(code)
    return code[match.start() : end]


def test_h3_preprocess_threads_the_descriptors_into_the_encode() -> None:
    """Asserted on the AST: a docstring or log line mentioning ``descriptors`` is not the call."""
    tree = ast.parse(_FNS.read_text(encoding="utf-8"))
    stage = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "h3_preprocess"
    )
    calls = [
        node
        for node in ast.walk(stage)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", ""))
        == "encode_h3_reference_latents"
    ]
    assert calls, "h3_preprocess no longer encodes references — re-target this guard"
    for call in calls:
        supplied = {kw.arg for kw in call.keywords}
        assert "descriptors" in supplied, (
            "the encode must be handed the resolved descriptors; without them the cache carries "
            "sizes but no identity and the trainer cannot read it (D-10-DEF-2)"
        )
        assert "explicit_manifest" in supplied, (
            "issue #52: the encode must be handed the row's provenance too; without it the cache "
            "cannot record whether its pool was explicit-manifest or rotating, and "
            "H3RefStrategy's mismatch refusal can never target the right rows"
        )


def test_the_descriptor_argument_is_required_never_defaulted() -> None:
    """A default would let a descriptor-less payload be written again, silently."""
    import inspect

    parameter = inspect.signature(encode_h3_reference_latents).parameters["descriptors"]
    assert parameter.default is inspect.Parameter.empty, (
        "descriptors must be REQUIRED: defaulting it to None/() re-creates D-10-DEF-2 — a full "
        "metered pre-encode that writes a cache the training stage refuses to read"
    )
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_nothing_in_the_resolver_defaults_kind_or_subject_id() -> None:
    """A fallback literal is how a guessed kind gets in. There must not be one.

    ``entry.get("kind", kind)`` is legal — ``kind`` is the value READ from the manifest key. A
    string literal as the fallback is not.
    """
    source = _stage_source("_h3_reference_entry")
    assert not re.search(r'\.get\(\s*"kind"\s*,\s*"', source), (
        "kind must never fall back to a string literal — a guessed kind silently reorders the "
        "references, and the shared rotary clock makes a reordered set a different request"
    )
    assert not re.search(r'\.get\(\s*"subject_id"\s*,', source), (
        "subject_id must never be defaulted — only the manifest can join a reference file to its "
        "identity label"
    )
    assert not re.search(r'\.stem|\.name\b', source), (
        "deriving subject_id from a filename is both a semantic guess (it claims every file is a "
        "different subject) and a client-property leak"
    )
