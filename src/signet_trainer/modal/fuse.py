"""In-Outpainting IC-LoRA fuse — the config-preserving pre-merge into the dev base (GATE-SPEC rev 2).

Ports the prior production project's PATCHED ``_FUSE_SCRIPT`` ([precedent]
``<prior-project>/_train/modal/`` Modal trainer — ``fuse_inoutpaint`` ~L231, the config-preserving
save ~L180-191, ``fuse_keycheck`` ~L196-227, ``embed_config`` ~L769-807) into a signet module.
Fuses the gated ``Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting`` adapter into the LTX-2.3 dev
base at strength 1.0, producing the FROZEN inpainting scaffold the inpaint LoRA trains on top of.

THE landmine this module exists to never repeat ([precedent] prior-project BUGLOG #5): ltx-core reads
the model config from the safetensors EMBEDDED metadata (``meta['config']``), not an external
file. The prior project's first fuse overwrote that metadata and dropped ``config`` -> the builder defaulted
every connector to gemma's 3840, mismatching the checkpoint's native dims (video 4096 /
audio 2048) — the fused base was unloadable. The patched fuse preserves the base's
``meta['config']`` verbatim; ``verify_fused_metadata`` below is the pre-dispatch gate that proves
it stuck (signet has ZERO ``embed_config`` repair mitigation — the fuse must be right first time).

Second landmine ([precedent] prior-project BUGLOG #6): do NOT audio-strip the fused base — loading an
audio-stripped checkpoint meta-device-crashes the trainer. ``verify_fused_metadata`` asserts the
audio weights are intact.

Key-space fact ([precedent], validated by the prior project's green fuse smoke): the base checkpoint's
raw state-dict keys are ``model.diffusion_model.*`` while the adapter ships ``diffusion_model.*``
— so the adapter keys get ``model.`` PREPENDED before the fuse. (Upstream's
``LTXV_LORA_COMFY_RENAMING_MAP`` goes the OTHER way — strips ``diffusion_model.`` for loading
into a bare transformer module — and does not apply to this raw-state-dict fuse path.)

Fuse math is CANONICAL, not re-derived: ``ltx_core.loader.apply_loras`` +
``LoraStateDictWithStrength`` + ``StateDict`` ([canonical] upstream, verified present at signet's
pinned SHA d6053703 via ``gh api`` — ``packages/ltx-core/src/ltx_core/loader/{__init__,fuse_loras,
primitives}.py``; ``apply_loras(model_sd, [LoraStateDictWithStrength(state_dict, strength)])``
returns a NEW StateDict, leaving the input tensors untouched — which is what makes the
changed-tensor probe below free).

Modal wiring (INTEGRATOR — this module deliberately does NOT import ``modal`` and declares no
``@app.function``; the integrator wires a ``fuse`` mode through the entrypoint gate):

  * Image: ``gpu_image`` (needs torch + ltx-core at the pinned SHA) but run CPU-ONLY — the fuse
    is a state-dict merge, NO ``gpu=`` (prior-project precedent: CPU + ``memory=131072``). The default
    ``apply_loras`` path materializes a SECOND full state dict (~2x 44GB) — 128 GiB RAM is
    load-bearing, not padding.
  * Volumes: ``WEIGHTS_MOUNT`` — the dev base lives at ``WEIGHTS_DIR / DEFAULT_BASE_FILENAME``
    and the fused output is written next to it (``WEIGHTS_DIR / DEFAULT_FUSED_FILENAME``, a base
    variant, NOT a checkpoint). Caller MUST ``weights_vol.commit()`` after ``fuse_inoutpaint``
    returns (commit-or-vanish) — this module cannot (no modal import).
  * Secrets ([precedent] prior-project GATED pattern): the adapter repo is HF-GATED — the prior project attached
    BOTH ``modal.Secret.from_name("hf-gated-secret")`` (a token whose HF account accepted the
    In-Outpainting license) AND ``modal.Secret.from_name("my-huggingface-secret")``; mirror that
    list (``GATED_HF_SECRET_NAME`` below). ``huggingface_hub`` picks the token up from the
    injected ``HF_TOKEN`` env var.
  * Naming: ``DEFAULT_FUSED_FILENAME`` intentionally contains no "distilled" so
    ``models.loader.base_variant_of`` classifies it as the dev family (GATE-SPEC rev 2) and the
    EXPECTED_* gate applies unchanged (fusing changes VALUES, never names/shapes).

Heavy imports (torch / safetensors / huggingface_hub / ltx_core) are FUNCTION-LOCAL — the
deliberate Anti-Pattern-6 exception style of ``models/loader.py``: importing this module for the
pure key/metadata logic (tests, CPU gates) needs stdlib only.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterable, Mapping
from typing import Any

# --------------------------------------------------------------------------------------------------
# Constants (config-first: all overridable through function parameters; these are the documented
# house defaults, [precedent] the prior project unless tagged otherwise).
# --------------------------------------------------------------------------------------------------

#: The gated official In-Outpainting IC-LoRA ([precedent] prior-project fuse_inoutpaint).
IN_OUTPAINT_LORA_REPO = "Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting"
IN_OUTPAINT_LORA_FILENAME = "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"

#: Modal secret name carrying an HF token with gated access to the In-Outpainting repo
#: ([precedent] prior-project ``GATED = [Secret.from_name("hf-gated-secret"), Secret.from_name(
#: "my-huggingface-secret")]``). The integrator attaches BOTH this and the regular HF secret.
GATED_HF_SECRET_NAME = "hf-gated-secret"

#: The dev base filename under WEIGHTS_DIR (matches fns.py's train/sample checkpoint_path).
DEFAULT_BASE_FILENAME = "ltx-2.3-22b-dev.safetensors"
#: Fused output filename ([precedent] prior-project out_name + ".safetensors"). No "distilled" in the
#: name -> ``base_variant_of`` resolves it as the dev family (GATE-SPEC rev 2).
DEFAULT_FUSED_FILENAME = "ltx-2.3-22b-inoutpaint-fused.safetensors"

#: Base keys are ``model.diffusion_model.*``; the adapter ships ``diffusion_model.*`` -> prepend.
LORA_KEY_PREFIX = "model."

#: Fuse provenance markers written into the output metadata ([precedent] prior-project _FUSE_SCRIPT).
FUSED_MARKER = "in-outpainting-0.9"
FUSED_MODEL_VERSION = "2.3"

#: [precedent] prior-project fuse strength — the scaffold is fused at full strength.
DEFAULT_FUSE_STRENGTH = 1.0


# --------------------------------------------------------------------------------------------------
# Pure key-space logic (stdlib-only; CPU-tested in tests/test_fuse_logic.py).
# --------------------------------------------------------------------------------------------------


def prepend_model_prefix(lora_sd: Mapping[str, Any]) -> dict[str, Any]:
    """Rename adapter keys ``diffusion_model.*`` -> ``model.diffusion_model.*``.

    [precedent] prior-project _FUSE_SCRIPT: ``{("model." + k): v for k, v in load_file(...).items()}``.
    Idempotent hardening on top of the blind prepend: a key already carrying the ``model.``
    prefix is passed through unchanged (re-running the transform can never double-prefix).
    Raises on a rename collision (two inputs mapping to one output would silently drop a tensor).
    """
    renamed: dict[str, Any] = {}
    for key, value in lora_sd.items():
        new_key = key if key.startswith(LORA_KEY_PREFIX) else LORA_KEY_PREFIX + key
        if new_key in renamed:
            raise ValueError(
                f"[fuse] key-prefix collision: {new_key!r} produced twice (input carried both "
                f"prefixed and unprefixed forms) — refusing to silently drop a tensor."
            )
        renamed[new_key] = value
    return renamed


def lora_target_weight_keys(lora_keys: Iterable[str]) -> set[str]:
    """Derive the base ``.weight`` keys a LoRA state dict would fuse into.

    Mirrors [canonical] upstream ``fuse_loras._affected_weight_keys`` at the pinned SHA exactly
    (``.lora_A.weight`` -> ``.weight``) so the pre-fuse keycheck and the actual fuse math agree
    on what a "target" is — single source of truth with ``apply_loras``.
    """
    suffix = ".lora_A.weight"
    return {k[: -len(suffix)] + ".weight" for k in lora_keys if k.endswith(suffix)}


def split_fuse_targets(
    renamed_lora_keys: Iterable[str], base_keys: Iterable[str]
) -> tuple[list[str], list[str]]:
    """Split the renamed adapter's derived target weights into (hits, misses) vs the base keys.

    ``hits`` = targets present in the base (will actually fuse); ``misses`` = derived targets
    absent from the base (a non-empty miss list with zero hits is the bad-rename signature
    the prior project's fuse ABORTed on). Both sorted for stable logs/tests.
    """
    targets = lora_target_weight_keys(renamed_lora_keys)
    base = set(base_keys)
    hits = sorted(t for t in targets if t in base)
    misses = sorted(t for t in targets if t not in base)
    return hits, misses


def build_fused_metadata(
    base_metadata: Mapping[str, str] | None,
    *,
    fused_marker: str = FUSED_MARKER,
    model_version: str = FUSED_MODEL_VERSION,
    require_config: bool = True,
) -> dict[str, str]:
    """Build the fused checkpoint's metadata, PRESERVING the base's ``meta['config']``.

    [precedent] prior-project BUGLOG #5 / the patched _FUSE_SCRIPT (~L180-191): dropping ``config``
    makes the fused base unloadable (every connector defaults to gemma's 3840 vs the native
    video 4096 / audio 2048 dims). The provenance markers (``fused`` / ``model_version``) match
    the precedent's byte-for-byte.

    ``require_config=True`` ([house] fail-fast, deliberately STRICTER than the prior project's
    warn-and-continue): a base without embedded config would produce a 44GB fused file signet has
    no ``embed_config`` repair path for — raise BEFORE any heavy work instead. The prior project's own
    ``embed_config`` repair tool applied the same stop-if-missing guard to the dev base.
    """
    metadata = {"fused": fused_marker, "model_version": model_version}
    config = (base_metadata or {}).get("config")
    if config:
        metadata["config"] = config
    elif require_config:
        raise RuntimeError(
            "[fuse] base checkpoint carries NO embedded 'config' metadata to preserve — a fused "
            "base without meta['config'] is unloadable (connectors default to gemma 3840 vs "
            "native video 4096 / audio 2048 — prior-project BUGLOG #5) and signet has no embed_config "
            "repair path. Refusing to fuse; check the base checkpoint."
        )
    return metadata


# --------------------------------------------------------------------------------------------------
# Header-only safetensors reads (stdlib struct+json — no safetensors/torch import needed;
# [precedent] prior-project embed_config's cheap header read). Safe on the 44GB file: reads only the
# 8-byte length prefix + the JSON header, never a tensor.
# --------------------------------------------------------------------------------------------------


def read_safetensors_header(path: str) -> tuple[dict[str, str], dict[str, dict]]:
    """Return ``(metadata, tensor_entries)`` from a safetensors file header (no tensor loads).

    ``metadata`` is the ``__metadata__`` dict (``{}`` if absent); ``tensor_entries`` maps each
    tensor name to its header entry (``{"dtype": ..., "shape": [...], "data_offsets": [...]}``).
    """
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    metadata = header.pop("__metadata__", None) or {}
    return metadata, header


def _audio_keys(tensor_names: Iterable[str]) -> set[str]:
    """The audio-branch weight keys ([precedent] prior-project make_vonly_base's 'audio' in k.lower())."""
    return {k for k in tensor_names if "audio" in k.lower()}


def verify_fused_metadata(
    fused_path: str,
    base_path: str | None = None,
    *,
    require_audio: bool = True,
) -> dict[str, Any]:
    """Pre-dispatch gate: assert the fused base is structurally sound. Header-only, CPU, ~free.

    Runs BEFORE any metered dispatch that would load the fused base (GATE-SPEC rev 2: "verify the
    fused file carries embedded config metadata before any metered dispatch" — signet has zero
    ``embed_config`` mitigation if it's missing). Asserts:

      1. ``meta['config']`` present + non-empty ([precedent] prior-project BUGLOG #5 — the
         3840-vs-2048/4096 connector crash).
      2. Audio weights intact (do-NOT-audio-strip, [precedent] prior-project BUGLOG #6 meta-device
         crash; disable via ``require_audio=False`` only for a deliberately video-only file).
      3. When ``base_path`` is given: identical tensor NAMES + shapes + dtypes vs the base
         (fusing changes values only — this is what keeps signet's EXPECTED_* arch gate valid on
         the fused file), and the preserved ``config`` string matches the base's byte-for-byte.

    Returns a summary dict; raises ``RuntimeError`` naming the exact failure otherwise.
    """
    metadata, tensors = read_safetensors_header(fused_path)

    config = metadata.get("config")
    if not config:
        raise RuntimeError(
            f"[fuse][verify] {fused_path} carries NO embedded 'config' metadata — the fused base "
            "is unloadable (connectors default to gemma 3840 vs native video 4096 / audio 2048; "
            "prior-project BUGLOG #5). Do NOT dispatch any metered run against this file; re-fuse with "
            "the config-preserving path."
        )

    audio = _audio_keys(tensors)
    if require_audio and not audio:
        raise RuntimeError(
            f"[fuse][verify] {fused_path} has ZERO audio-branch weights — an audio-stripped base "
            "meta-device-crashes the trainer (prior-project BUGLOG #6, do-NOT-audio-strip). Refusing."
        )

    summary: dict[str, Any] = {
        "fused_path": fused_path,
        "n_keys": len(tensors),
        "n_audio_keys": len(audio),
        "config_chars": len(config),
        "fused_marker": metadata.get("fused"),
    }

    if base_path is not None:
        base_metadata, base_tensors = read_safetensors_header(base_path)
        missing = sorted(set(base_tensors) - set(tensors))
        extra = sorted(set(tensors) - set(base_tensors))
        if missing or extra:
            raise RuntimeError(
                f"[fuse][verify] tensor NAME set diverged from the base (fusing must change "
                f"values only): {len(missing)} missing (e.g. {missing[:3]}), {len(extra)} extra "
                f"(e.g. {extra[:3]}). The EXPECTED_* gate would be invalid — refusing."
            )
        mismatched = sorted(
            k
            for k in base_tensors
            if (
                tensors[k].get("shape") != base_tensors[k].get("shape")
                or tensors[k].get("dtype") != base_tensors[k].get("dtype")
            )
        )
        if mismatched:
            k = mismatched[0]
            raise RuntimeError(
                f"[fuse][verify] {len(mismatched)} tensor(s) changed shape/dtype vs the base "
                f"(first: {k!r} base={base_tensors[k]['dtype']}{base_tensors[k]['shape']} fused="
                f"{tensors[k]['dtype']}{tensors[k]['shape']}) — fusing must be value-only. Refusing."
            )
        base_config = base_metadata.get("config")
        if base_config and config != base_config:
            raise RuntimeError(
                "[fuse][verify] embedded 'config' metadata DIFFERS from the base's — the fuse "
                "must preserve it verbatim (prior-project BUGLOG #5). Refusing."
            )
        summary["base_compared"] = True

    return summary


# --------------------------------------------------------------------------------------------------
# Changed-tensor diff (torch import function-local; CPU-testable with tiny tensors).
# --------------------------------------------------------------------------------------------------


def diff_changed_keys(
    base_sd: Mapping[str, Any],
    fused_sd: Mapping[str, Any],
    candidate_keys: Iterable[str],
) -> list[str]:
    """Return the candidate keys whose tensors actually CHANGED between base and fused dicts.

    The fuse_keycheck-style probe core ([precedent] the prior project's single-probe "weight CHANGED by
    fuse" check, widened to every expected target): a zero-length result after a fuse means the
    rename/merge silently did nothing. Free after ``apply_loras``'s default path, which returns a
    NEW StateDict and never mutates the input tensors ([canonical] verified at the pin:
    ``_bf16_fuse`` adds the weight into the freshly-aggregated delta, not in-place).
    """
    import torch  # noqa: PLC0415 — function-local (Anti-Pattern-6 exception style)

    changed = []
    for key in candidate_keys:
        if key not in base_sd or key not in fused_sd:
            continue
        if not torch.equal(base_sd[key], fused_sd[key]):
            changed.append(key)
    return changed


# --------------------------------------------------------------------------------------------------
# Modal-side plain functions (heavy imports function-local). The integrator wires these to an
# entrypoint mode + @app.function; NO metered dispatch happens here.
# --------------------------------------------------------------------------------------------------


def download_inoutpaint_lora(
    repo_id: str = IN_OUTPAINT_LORA_REPO,
    filename: str = IN_OUTPAINT_LORA_FILENAME,
) -> str:
    """Download the gated In-Outpainting adapter; returns the local path.

    Requires an HF token with accepted gated access in the environment (``HF_TOKEN`` — injected
    by the ``hf-gated-secret`` + ``my-huggingface-secret`` pair, [precedent] prior-project GATED).
    """
    from huggingface_hub import hf_hub_download  # noqa: PLC0415 — function-local heavy import

    return hf_hub_download(repo_id, filename)


def fuse_keycheck(
    base_path: str,
    lora_path: str | None = None,
    *,
    lora_repo: str = IN_OUTPAINT_LORA_REPO,
    lora_filename: str = IN_OUTPAINT_LORA_FILENAME,
) -> dict[str, Any]:
    """CHEAP pre-fuse probe ([precedent] prior-project fuse_keycheck): header-only key matching.

    Confirms the ``model.``-prepended adapter keys land on real base weights BEFORE paying for
    the heavy fuse — reads ONLY the two file headers (no 44GB load, no torch). Raises if zero
    targets hit (the bad-rename signature). Returns ``{n_lora_keys, n_targets, n_hits, hits,
    misses}``.
    """
    if lora_path is None:
        lora_path = download_inoutpaint_lora(lora_repo, lora_filename)

    _, lora_tensors = read_safetensors_header(lora_path)
    _, base_tensors = read_safetensors_header(base_path)

    renamed = prepend_model_prefix(dict.fromkeys(lora_tensors))
    hits, misses = split_fuse_targets(renamed, base_tensors)
    print(
        f"[fuse][keycheck] lora keys: {len(lora_tensors)} | derived targets: "
        f"{len(hits) + len(misses)} | present in base: {len(hits)} | missing: {len(misses)}",
        flush=True,
    )
    if not hits:
        raise RuntimeError(
            "[fuse][keycheck] ZERO renamed adapter targets matched the base — bad rename "
            "(expected base keys 'model.diffusion_model.*', adapter 'diffusion_model.*' + "
            "'model.' prepend). NOT safe to fuse; aborting before any heavy load."
        )
    return {
        "n_lora_keys": len(lora_tensors),
        "n_targets": len(hits) + len(misses),
        "n_hits": len(hits),
        "hits": hits,
        "misses": misses,
    }


def fuse_inoutpaint(
    base_path: str,
    out_path: str,
    *,
    strength: float = DEFAULT_FUSE_STRENGTH,
    lora_repo: str = IN_OUTPAINT_LORA_REPO,
    lora_filename: str = IN_OUTPAINT_LORA_FILENAME,
    lora_path: str | None = None,
    require_config: bool = True,
) -> dict[str, Any]:
    """Fuse the In-Outpainting IC-LoRA into the dev base, PRESERVING ``meta['config']``.

    The ported patched prior-project fuse ([precedent] _FUSE_SCRIPT + fuse_inoutpaint), in order:

      1. Header-only pre-checks BEFORE any heavy load: build the config-preserving metadata
         (fail-fast on a config-less base, [house]) + the fuse_keycheck key-space probe.
      2. Load the adapter, prepend ``model.`` to its ``diffusion_model.*`` keys.
      3. Load the full base (~44GB — CPU, needs ~128 GiB RAM: ``apply_loras``'s default path
         materializes a second full dict; [precedent] the prior project ran memory=131072).
      4. ``apply_loras`` at ``strength`` ([canonical] ltx_core.loader at pin d6053703).
      5. Probe: assert N > 0 target tensors ACTUALLY changed vs the base (catches a silent
         no-op merge; the prior project's single-probe check, widened to every expected target).
      6. ``save_file`` WITH the base's ``meta['config']`` preserved (prior-project BUGLOG #5).
      7. Self-check the written file with ``verify_fused_metadata`` (config + audio + value-only
         vs the base) so a broken artifact can never look done.

    Returns a summary dict. The CALLER commits the Volume (``weights_vol.commit()``) — this
    module never imports modal. NO metered dispatch here: this is a plain function the
    entrypoint-gated integrator wiring invokes.
    """
    # Function-local heavy imports (Anti-Pattern-6 exception style, models/loader.py).
    import torch  # noqa: PLC0415
    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    from ltx_core.loader import (  # noqa: PLC0415 — [canonical] verified at pin d6053703
        LoraStateDictWithStrength,
        StateDict,
        apply_loras,
    )

    def _mk_sd(sd: dict) -> StateDict:
        # [precedent] prior-project mk_sd — StateDict(sd, device, size, dtype_set) at the pin.
        size = sum(t.numel() * t.element_size() for t in sd.values())
        return StateDict(sd, torch.device("cpu"), size, {t.dtype for t in sd.values()})

    # ── (1) header-only pre-checks BEFORE the heavy loads (fail-fast, zero wasted RAM/time) ─────
    base_metadata, _base_header = read_safetensors_header(base_path)
    fused_metadata = build_fused_metadata(base_metadata, require_config=require_config)
    if "config" in fused_metadata:
        print(
            f"[fuse] preserving base config metadata ({len(fused_metadata['config'])} chars) — "
            "prior-project BUGLOG #5 guard.",
            flush=True,
        )
    else:
        print("[fuse] WARNING: base carries no 'config' metadata to preserve!", flush=True)

    if lora_path is None:
        lora_path = download_inoutpaint_lora(lora_repo, lora_filename)
    keycheck = fuse_keycheck(base_path, lora_path)
    hits = keycheck["hits"]

    # ── (2) load + rename the adapter (diffusion_model.* -> model.diffusion_model.*) ────────────
    lora_renamed = prepend_model_prefix(load_file(lora_path))

    # ── (3) load the full base (the 44GB state dict — CPU, big-RAM) ─────────────────────────────
    print("[fuse] loading full base state dict...", flush=True)
    base_raw = load_file(base_path)
    print(f"[fuse] base keys: {len(base_raw)} | lora keys: {len(lora_renamed)}", flush=True)

    # ── (4) the canonical fuse at ``strength`` ──────────────────────────────────────────────────
    fused = apply_loras(
        _mk_sd(base_raw), [LoraStateDictWithStrength(_mk_sd(lora_renamed), strength)]
    )

    # ── (5) changed-tensor probe: N targets must ACTUALLY differ from the base ──────────────────
    # apply_loras's default path returns fresh tensors and never mutates base_raw ([canonical]
    # at the pin), so this diff needs no pre-cloning.
    changed = diff_changed_keys(base_raw, fused.sd, hits)
    print(
        f"[fuse] changed-tensor probe: {len(changed)}/{len(hits)} expected targets changed",
        flush=True,
    )
    if not changed:
        raise RuntimeError(
            f"[fuse] fuse was a SILENT NO-OP — 0 of {len(hits)} expected target tensors changed "
            "vs the base (prefix mismatch or zero-delta adapter). Not saving a fake fused base."
        )
    if len(changed) != len(hits):
        print(
            f"[fuse] WARNING: {len(hits) - len(changed)} expected target(s) did NOT change "
            "(zero LoRA delta?) — inspect before trusting the scaffold.",
            flush=True,
        )

    # ── (6) save WITH the preserved config metadata (the BUGLOG-#5 fix) ─────────────────────────
    print(f"[fuse] saving fused base -> {out_path}", flush=True)
    save_file(fused.sd, out_path, metadata=fused_metadata)

    # ── (7) self-verify the written artifact (config + audio intact + value-only vs base) ───────
    verification = verify_fused_metadata(out_path, base_path=base_path)
    print(f"[fuse] FUSED_DONE -> {out_path} | verify: {verification}", flush=True)

    return {
        "out_path": out_path,
        "strength": strength,
        "n_base_keys": len(base_raw),
        "n_lora_keys": len(lora_renamed),
        "n_expected_targets": len(hits),
        "n_changed": len(changed),
        "verification": verification,
    }
