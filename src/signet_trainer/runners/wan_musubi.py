"""The musubi-tuner Wan 2.1 RECIPE and its three subprocess invocations. PURE stdlib.

WHAT THIS IS. ``runners/musubi_toml.py`` translates the signet manifest into musubi's *dataset*
dialect; this module translates it into musubi's *command line*. Both directions are needed because
the runner owns caching end to end: ``wan_cache_latents.py`` and
``wan_cache_text_encoder_outputs.py`` read the dataset TOML directly and ``wan_train_network.py``
reads it a third time (``docs/source-methods/musubi-wan21/train_kohya.py:109,121,140``).

⛔ STDLIB ONLY — NOT MERELY THE PACKAGE CONVENTION, A HARD REQUIREMENT OF THE IMAGE THIS RUNS ON.
Its sibling ``musubi_toml.py`` may import ``config/sources.py`` (and therefore pydantic v2) because
it only ever runs LOCALLY. This module is imported INSIDE the musubi container, where pydantic v2
does not exist and cannot:

    train_kohya.py:37   .pip_install("pydantic==1.10.13", "hf_transfer")
    pyproject.toml      [project.dependencies]  pydantic>=2.10.4

The house recipe pins **pydantic 1.10.13** into the image, AFTER musubi's own
``requirements.txt``, and signet's ``config/load.load_config_from_text`` needs pydantic **v2**. One
interpreter cannot carry both. That single line is why ``modal/fns.wan_train`` is a
THREADED-PARAMETER stage (the ``h3_preprocess`` shape) rather than a ``config_yaml: str`` stage (the
``train`` / ``h3_train`` / ``qwen_edit_train`` shape): it never loads a ``SignetConfig``
in-container, so it never touches pydantic at all. Adding a pydantic import anywhere in this
module's closure breaks the stage inside a metered container, at the first statement, for a reason
that is invisible from a laptop where both versions resolve fine.

[UNVERIFIED] WHY musubi wants pydantic 1.x is not established here — the pin is TRANSCRIBED from the
house recipe, not re-derived, and this pass performs zero downloads. What IS established is that the
two pins are mutually exclusive, which is all the design decision needs.

THE RECIPE IS LOCKED HERE, NOT IN THE CONFIG, and that is the ``QWEN_EDIT_RENDER_RECIPE`` precedent
rather than a shortcut. Every value in :data:`WAN_MUSUBI_RECIPE` is transcribed line-for-line from
``train_kohya.py:129-160`` — the one Wan 2.1 run this project has a working record of. They are
recipe terms in the same sense that ``quantize_qwen_edit``'s qfloat8 is one: settled by the method,
not knobs an operator turns per round. What DOES vary between rounds — the dataset — is exactly what
``runners/musubi_toml.render_from_config`` renders, which is what makes "only the dataset changed
between rounds" a structural property of this stage instead of a discipline.

⚠ ``discrete_flow_shift = 7.0`` is a REAL, REQUIRED parameter of this family and is why this module
carries the ``wan_`` filename prefix: ``tests/test_no_wan_params.py`` was rescoped by FAMILY on
2026-08-09 so that a family-prefixed module may say ``shift`` but may only pin ITS OWN value (7.0 for
``wan_``, never Wan's 4.0 in a Qwen sampler, never a stray default). The prefix is what puts this
file under that stricter value-level rule instead of outside every rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

__all__ = [
    "WAN_CACHE_LATENTS_SCRIPT",
    "WAN_CACHE_TEXT_ENCODER_SCRIPT",
    "WAN_COMPONENT_CONFIG_FIELDS",
    "WAN_MUSUBI_RECIPE",
    "WAN_TRAIN_NETWORK_SCRIPT",
    "WanComponents",
    "WanMusubiRecipe",
    "wan_cache_latents_argv",
    "wan_cache_text_encoder_argv",
    "wan_clips_per_media",
    "wan_resolve_component_ids",
    "wan_stage_argv",
    "wan_train_network_argv",
]

# --------------------------------------------------------------------------------------------------
# The three upstream entry points, by NAME. Relative, because every one is invoked with the
# musubi-tuner checkout as CWD (train_kohya.py's image does ``.workdir("musubi-tuner")``).
# --------------------------------------------------------------------------------------------------

WAN_CACHE_LATENTS_SCRIPT = "wan_cache_latents.py"
WAN_CACHE_TEXT_ENCODER_SCRIPT = "wan_cache_text_encoder_outputs.py"
WAN_TRAIN_NETWORK_SCRIPT = "wan_train_network.py"


@dataclass(frozen=True)
class WanMusubiRecipe:
    """The LOCKED Wan 2.1 t2v-14B LoRA recipe, transcribed from ``train_kohya.py:129-160``.

    Frozen and module-level for the same reason ``QWEN_EDIT_RENDER_RECIPE`` is: ONE declaration,
    read by the entrypoint (to print what the run will do) and by the container (to build the argv).
    A value threaded through the dispatch instead would be two numbers that can disagree, and the
    disagreement would be visible only in a log nobody diffs.
    """

    #: ``--task``. t2v-14B is the text-to-video 14B Wan 2.1 task. NOT i2v: train_kohya.py downloads
    #: the T5/CLIP encoders from the I2V repo (they are shared) but trains the T2V DiT.
    task: str = "t2v-14B"
    #: ``--network_module`` / ``--network_dim``. musubi's own Wan LoRA implementation.
    network_module: str = "networks.lora_wan"
    network_dim: int = 32
    #: ``--discrete_flow_shift``. The static flow shift for this family. See the ⚠ note in the module
    #: docstring: 7.0 is Wan's own value and the ONLY literal ``tests/test_no_wan_params.py`` admits
    #: from a ``wan_``-prefixed module.
    discrete_flow_shift: float = 7.0
    #: ``--learning_rate`` — a STRING, not a float, because argv is strings and ``str(2e-4)`` is
    #: ``'0.0002'``. Both parse the same; keeping the source spelling keeps a rendered command line
    #: diffable against the transcription it came from.
    learning_rate: str = "2e-4"
    optimizer_type: str = "adamw8bit"
    optimizer_args: str = "weight_decay=0.01"
    #: ``--max_train_epochs``. ⚠ musubi is EPOCH-driven. ``training.max_steps`` in the signet config
    #: is NOT read by the runner — it is signet's accounting basis for the cost line. The two are
    #: printed side by side by ``modal/cost.format_wan_batch_line`` precisely so the difference is
    #: visible rather than assumed away.
    max_train_epochs: int = 16
    save_every_n_epochs: int = 1
    mixed_precision: str = "bf16"
    #: ``--batch_size`` of the TEXT-ENCODER cache pass (train_kohya.py:123). Unrelated to a
    #: dataset's per-source ``batch_size``, which lives in the rendered TOML.
    text_encoder_cache_batch_size: int = 16
    #: ``accelerate launch --num_cpu_threads_per_process``.
    num_cpu_threads_per_process: int = 1


#: THE recipe. One instance, read from both sides of the dispatch.
WAN_MUSUBI_RECIPE = WanMusubiRecipe()


@dataclass(frozen=True)
class WanComponents:
    """The four Wan 2.1 weight components, as ids RELATIVE to the weights Volume mount.

    Ids, never absolute paths: the mount point is a CONTAINER fact (``modal/app.WEIGHTS_DIR``) and
    the ids are a CONFIG fact, so the container composes ``WEIGHTS_DIR / id`` and the local side
    never has to know where the Volume lands. Same split ``h3_preprocess`` uses for its four
    component ids.

    ⚠ These are the four files ``train_kohya.py:69-92`` obtains by ``hf_hub_download`` from two
    HARDCODED Hub repo ids at run time. signet does not do that: weights are resolved from the
    weights Volume so a run is reproducible, air-gappable and not silently re-downloading 30+ GiB
    per container. That difference is the whole reason :func:`wan_resolve_component_ids` exists.
    """

    dit: str
    vae: str
    t5: str
    clip: str


#: Which ``model:`` field carries each component, named ONCE so the entrypoint's gap check and the
#: resolver below cannot drift apart. ``clip`` maps to a field that DOES NOT EXIST on ``ModelConfig``
#: yet — that is the point, and :func:`wan_resolve_component_ids` refuses on it by name.
WAN_COMPONENT_CONFIG_FIELDS: dict[str, str] = {
    "dit": "model_id",
    "vae": "vae_id",
    "t5": "text_encoder_id",
    "clip": "clip_id",
}


def wan_resolve_component_ids(
    model_cfg: object, *, schema_defaults: Mapping[str, object] | None = None
) -> WanComponents:
    """Read the four component ids off a ``model:`` block. NOT LANDED — refuses, naming each gap.

    ``model_cfg`` is typed ``object`` and read by ``getattr`` with a default rather than annotated
    ``ModelConfig``, the ``_qwen_edit_render_request`` idiom: this module must stay importable on an
    interpreter that has no pydantic (see the module docstring), and the fields it wants do not all
    exist yet. The day they land, every caller starts seeing real values and nothing here changes.

    ⛔⛔ ``schema_defaults`` IS NOT OPTIONAL POLISH — IT CLOSES A SILENT WRONG-WEIGHTS HOLE. Two of
    the four fields are shared with the LTX family and carry LTX DEFAULTS, not ``None``:
    ``ModelConfig.text_encoder_id`` defaults to ``"gemma-3-12b-it"`` and ``model_id`` to the LTX
    checkpoint. A truthiness test alone therefore RESOLVES on a wan config that declared neither,
    and the stage would hand musubi ``--t5 /weights/gemma-3-12b-it`` — a Gemma directory where it
    expects a umT5 ``.pth``. That fails, expensively and confusingly, after the latent cache pass
    has already run; and the failure reads as a musubi bug rather than as a config that never named
    a text encoder. Passing ``{field: ModelConfig.model_fields[field].default}`` lets this refuse an
    INHERITED value as sharply as a missing one, and DERIVING those defaults from the schema (rather
    than restating them here) is what keeps the check true when a default changes. Omitting the
    argument keeps the older, weaker behaviour and is why the parameter is keyword-only and
    documented rather than silently defaulted away.

    FOUR GAPS OF THREE DIFFERENT KINDS, reported together rather than as one blanket "weights are
    not wired", because each needs a different edit:

      * ``model.model_id`` / ``model.text_encoder_id`` — fields that EXIST and are shared with LTX,
        so on a wan config they are INHERITED rather than unset. Declaring them is a config edit.
      * ``model.vae_id`` — a field that exists and is REFUSED on this family.
        ``config/schema._FAMILY_ONLY_MODEL_IDS`` maps it to ``frozenset({"h3", "qwen_edit"})``, so a
        ``family: wan`` config that sets it fails at config LOAD with *"that ID is only read under
        family {...} and would be silently ignored here"*. That was correct while the field had no
        consumer on this family; it now has one. WHAT LANDS IT: add ``"wan"`` to that entry — the
        same one-line change ``_qwen_edit_config_gaps`` gap (1) asks for, for the same reason.
      * ``model.clip_id`` — a field that does NOT EXIST AT ALL. Wan 2.1 needs an open-CLIP
        XLM-RoBERTa-Large ViT-H/14 encoder ALONGSIDE the umT5 text encoder
        (``train_kohya.py:78-82,139``); no signet family has ever had two text-side encoders, so
        ``ModelConfig`` has no slot for the second. WHAT LANDS IT: a ``clip_id: str | None`` field on
        ``ModelConfig``, fenced to ``{"wan"}`` in ``_FAMILY_ONLY_MODEL_IDS`` so no other family can
        set a knob it does not read.

    ⛔ NOTHING IS GUESSED HERE. Composing the CLIP path from ``text_encoder_id`` is the one tempting
    shortcut and it is wrong in the way that costs money: musubi would load a umT5 checkpoint where
    it expects open-CLIP, inside a metered container, after the dataset cache pass has already run.

    Raises:
        NotImplementedError: until all four ids resolve to DECLARED values. The message separates
            the inherited ones from the absent ones and names what lands each.
    """
    defaults: Mapping[str, object] = schema_defaults or {}
    resolved: dict[str, str] = {}
    absent: list[str] = []
    inherited: list[str] = []
    for component, field in WAN_COMPONENT_CONFIG_FIELDS.items():
        value = getattr(model_cfg, field, None)
        if not value:
            absent.append(f"{component} (model.{field})")
        elif field in defaults and value == defaults[field]:
            inherited.append(f"{component} (model.{field} is still its schema default {value!r})")
        else:
            resolved[component] = str(value)
    if not absent and not inherited:
        return WanComponents(**resolved)
    raise NotImplementedError(
        "wan_resolve_component_ids: the Wan 2.1 weight components are not resolvable from this "
        f"model: block — {len(absent)} absent {absent}, {len(inherited)} INHERITED FROM THE LTX "
        f"DEFAULT {inherited}. An inherited value is refused as firmly as a missing one: "
        "model.text_encoder_id defaults to LTX's Gemma encoder, so accepting it would hand musubi "
        "--t5 <weights>/gemma-3-12b-it where it expects a umT5 checkpoint, after the latent cache "
        "pass has already run. train_kohya.py:69-92 obtains all four components by hf_hub_download "
        "from two HARDCODED Hub repo ids at run time; signet resolves them from the weights Volume "
        "instead, so each needs a DECLARED config field. WHAT LANDS THEM: declare model.model_id "
        "(the Wan DiT safetensors) and model.text_encoder_id (the umT5 .pth) explicitly. "
        "model.vae_id EXISTS but config/schema._FAMILY_ONLY_MODEL_IDS fences it to "
        "{'h3', 'qwen_edit'}, so add 'wan' to that entry. model.clip_id does not exist at all — Wan "
        "needs an open-CLIP encoder BESIDE the umT5 one and no signet family has ever carried two, "
        "so it is a new field on ModelConfig, fenced to {'wan'}. Do NOT compose the CLIP path from "
        "text_encoder_id: musubi would load umT5 where it expects open-CLIP, in a metered "
        "container, after the cache pass has already run."
    )


# --------------------------------------------------------------------------------------------------
# The three subprocess invocations — transcribed, then parameterised only where signet differs
# --------------------------------------------------------------------------------------------------


def wan_cache_latents_argv(*, dataset_config: str, vae: str, clip: str) -> list[str]:
    """Stage 1 of 3 — ``wan_cache_latents.py`` (``train_kohya.py:105-114``), verbatim.

    ``--vae_cache_cpu`` is carried because the transcription carries it: it keeps the VAE's cache on
    CPU RAM rather than VRAM, which is what makes the 14B pass fit beside the encoder.
    """
    return [
        "python",
        WAN_CACHE_LATENTS_SCRIPT,
        "--dataset_config", dataset_config,
        "--vae", vae,
        "--clip", clip,
        "--vae_cache_cpu",
    ]


def wan_cache_text_encoder_argv(
    *, dataset_config: str, t5: str, recipe: WanMusubiRecipe = WAN_MUSUBI_RECIPE
) -> list[str]:
    """Stage 2 of 3 — ``wan_cache_text_encoder_outputs.py`` (``train_kohya.py:116-126``), verbatim.

    ⚠ The caption granularity trap lives HERE, not in this argv. This pass caches ONE text embedding
    per CAPTION FILE, while the latent cache above is per CLIP — so a caption naming a moment
    ("reaching toward each other") is false for a ``uniform`` window starting at frame 75. That is
    what ``SourceSpec.caption_extension``'s per-source override exists for; nothing in this command
    line can detect it.
    """
    return [
        "python",
        WAN_CACHE_TEXT_ENCODER_SCRIPT,
        "--dataset_config", dataset_config,
        "--t5", t5,
        "--batch_size", str(recipe.text_encoder_cache_batch_size),
        "--fp8_t5",
    ]


def wan_train_network_argv(
    *,
    dataset_config: str,
    components: WanComponents,
    output_dir: str,
    output_name: str,
    seed: int,
    recipe: WanMusubiRecipe = WAN_MUSUBI_RECIPE,
) -> list[str]:
    """Stage 3 of 3 — ``accelerate launch wan_train_network.py`` (``train_kohya.py:128-161``).

    Transcribed flag for flag. Only five values are parameterised, and each for a stated reason:
    the four component paths and the dataset TOML (signet resolves them from Volumes rather than
    from a Hub download), the output directory and name (signet keys them on ``cfg.output_dir``, so
    a round's artifacts land where every other family's do), and the seed (``cfg.seed`` —
    D-NOHARDCODE; the transcription's literal 42 is the shipped config's value anyway, so this
    changes nothing about the reference run and makes a second one expressible).

    ⚠ ``--mixed_precision bf16`` appears TWICE and that is not a transcription slip: the first is
    ``accelerate launch``'s, the second is the training script's own. Dropping either changes what
    runs.
    """
    return [
        "accelerate", "launch",
        "--num_cpu_threads_per_process", str(recipe.num_cpu_threads_per_process),
        "--mixed_precision", recipe.mixed_precision,
        WAN_TRAIN_NETWORK_SCRIPT,
        "--task", recipe.task,
        "--dit", components.dit,
        "--vae", components.vae,
        "--t5", components.t5,
        "--clip", components.clip,
        "--dataset_config", dataset_config,
        "--sdpa",
        "--mixed_precision", recipe.mixed_precision,
        "--fp8_base",
        "--fp8_scaled",
        "--vae_cache_cpu",
        "--optimizer_type", recipe.optimizer_type,
        "--optimizer_args", recipe.optimizer_args,
        "--learning_rate", recipe.learning_rate,
        "--gradient_checkpointing",
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
        "--network_module", recipe.network_module,
        "--network_dim", str(recipe.network_dim),
        "--discrete_flow_shift", str(recipe.discrete_flow_shift),
        "--max_train_epochs", str(recipe.max_train_epochs),
        "--save_every_n_epochs", str(recipe.save_every_n_epochs),
        "--seed", str(seed),
        "--output_dir", output_dir,
        "--output_name", output_name,
    ]


def wan_stage_argv(
    *,
    dataset_config: str,
    components: WanComponents,
    output_dir: str,
    output_name: str,
    seed: int,
    recipe: WanMusubiRecipe = WAN_MUSUBI_RECIPE,
) -> tuple[tuple[str, list[str]], ...]:
    """The three stages, IN ORDER, as ``(label, argv)`` pairs.

    Returned as one ordered structure rather than three separate calls so the Modal stage can
    iterate and COUNT — a pass that is skipped then prints a zero instead of printing nothing, which
    is the difference between an auditable log and a log that looks fine because a line is absent.
    The ORDER is load-bearing: ``wan_train_network.py`` reads the caches the two passes above
    write, and musubi does not build them on demand.
    """
    return (
        (
            "cache_latents",
            wan_cache_latents_argv(
                dataset_config=dataset_config, vae=components.vae, clip=components.clip
            ),
        ),
        (
            "cache_text_encoder_outputs",
            wan_cache_text_encoder_argv(
                dataset_config=dataset_config, t5=components.t5, recipe=recipe
            ),
        ),
        (
            "train_network",
            wan_train_network_argv(
                dataset_config=dataset_config,
                components=components,
                output_dir=output_dir,
                output_name=output_name,
                seed=seed,
                recipe=recipe,
            ),
        ),
    )


# --------------------------------------------------------------------------------------------------
# Clip arithmetic — how much WORK one media file contributes, per source
# --------------------------------------------------------------------------------------------------


def wan_clips_per_media(
    extraction: str, target_frames: Sequence[int], frame_sample: int
) -> int | None:
    """How many clips ONE media file yields under ``extraction``; ``None`` when it is not knowable.

    Derived from ``ExtractionMode``'s own transcription of the upstream ``if/elif`` chain
    (``config/sources.py``). Takes plain values rather than a ``SourceSpec`` so this module keeps
    its stdlib-only, pydantic-free closure (see the module docstring).

    ``None`` IS THE HONEST ANSWER for two of the six modes, and returning it rather than a guess is
    the point:

      * ``chunk`` tiles the whole video and DROPS the remainder — the count is
        ``len(video) // target``, and the video length is on a Volume this process cannot read
        (Pitfall 1: no filesystem touch during validation).
      * ``slide`` walks windows at ``0, stride, 2*stride, ...`` until the video runs out — same
        dependency.

    The other four are arithmetic: ``head`` yields one clip per declared length, ``uniform`` yields
    ``frame_sample`` windows per declared length, ``full`` is one clip by definition, and an image
    contributes itself.
    """
    if extraction == "image":
        return 1
    if extraction == "full":
        return 1
    if extraction == "head":
        return len(target_frames)
    if extraction == "uniform":
        return frame_sample * len(target_frames)
    if extraction in ("chunk", "slide"):
        return None
    raise ValueError(
        f"wan_clips_per_media: unknown extraction {extraction!r}. The vocabulary is "
        f"ExtractionMode's (config/sources.py) and has exactly six members; a seventh name here "
        f"would be priced as something musubi's schema validator rejects."
    )
