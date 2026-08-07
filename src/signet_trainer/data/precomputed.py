"""data.precomputed — PrecomputedDataset: the loop-purity data path (TRAIN-03 / D-08).

Reads ``.precomputed/{latents,conditions}/<rel>.pt``, pairs them by relative path, and
returns ``{"latent_conditions": {...}, "text_conditions": {...}}``. Video-only:
``data_sources=["latents","conditions"]`` (enochiatron's ``audio_latents/`` source is
dropped — signet-trainer is video reference-control only).

Ported from ``Lightricks/LTX-2`` ``packages/ltx-trainer/src/ltx_trainer/datasets.py``
``PrecomputedDataset`` (RESEARCH.md Pattern 3). The upstream ``from ltx_trainer import
logger`` is deliberately replaced with stdlib ``logging`` to preserve import confinement.

IC-LoRA (07-08 / Pitfall 3): the class is source-count-agnostic — a THIRD
``reference_latents/`` source (the seg-map / reference-video latents) pairs by relative path
alongside ``latents`` + ``conditions`` with NO code change. ``_discover_samples`` pairs every
configured source by rel path, and the ``_VIDEO_LATENT_SOURCE_DIRS`` allowlist in ``__getitem__``
applies the SAME ``[C, F, H, W]`` normalization to ``reference_latents`` as to ``latents`` (they
share the VAE-encode shape). ``_get_expected_file_path``'s legacy ``conditions`` rename does NOT
misfire on ``reference_latents`` (the rename is gated on ``dir_name == "conditions"``). The 3-source
spec is wired by the caller (``modal/fns.py`` train step, via ``strategy.get_data_sources()``):
``{"latents": "latent_conditions", "conditions": "text_conditions",
"reference_latents": "ref_latent_conditions"}``. See ``tests/test_precomputed.py`` for the proof.

Inpaint ``video_masks`` source (Phase 9, GATE-SPEC rev 2): the signet-native mask encode
(``data/mask_encode.py``) writes one BARE float32 ``[F_lat, H_lat, W_lat]`` {0,1} tensor per sample
under ``video_masks/`` mirroring the ``latents/`` rel-path layout, so it registers as another
optional extra source exactly like ``reference_latents`` — paired by rel path with NO code change
here. CRITICAL: ``video_masks`` and ``audio_latents`` are NOT in the ``_VIDEO_LATENT_SOURCE_DIRS``
allowlist below, so their tensors are NOT routed through ``_normalize_video_latents`` — they pass
through UN-normalized (mask tensors are latent-GRID masks; a2v audio latents are AUDIO-VAE latents,
neither is a video VAE latent and neither carries the ``"latents"`` key the normalizer reads).
GATE-SPEC rev 2 item 4: the allowlist REPLACED the old ``"latent" in dir_name`` substring test
precisely because ``audio_latents`` contains the "latent" substring and would otherwise trip the
normalizer. Wired by the caller via ``modal/fns.py::_PRECOMPUTED_SOURCE_OUTPUT_KEYS``
(``"video_masks": "video_mask_conditions"``; ``"audio_latents": "audio_latent_conditions"``).
Regression proof: ``tests/test_mask_encode.py`` + ``tests/test_precomputed_audio_latents.py``.

The legacy patchified ``[seq_len, C]`` -> ``[C, F, H, W]`` auto-conversion uses ``einops``,
which is NOT a current dependency (signet's canonical ``process_dataset.py`` output is already
non-patchified). To keep this module importable today with zero new deps, ``einops`` is
imported LAZILY inside that legacy branch only — Phase-3 carry-forward: declare ``einops``
when (if) a legacy patchified cache must be read.

CRITICAL — Anti-Pattern 6 / TRAIN-03 / D-08:
    This module imports ONLY ``torch`` + stdlib (``pathlib``, ``concurrent.futures``,
    ``logging``) at module load (``einops`` lazily, in the legacy branch only). It MUST NOT
    import ``modal``, ``ltx_core``, ``ltx_trainer``, the VAE/Gemma loaders, or
    ``models/loader.py`` — importing ``signet_trainer.data.precomputed`` must NOT
    transitively pull any encoder. ``tests/test_loop_purity.py`` fails CI if it does.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

#: The precomputed cache dir name (FLAT layout — bucket is recorded inside the tensor
#: shape, NOT in the filename; RESEARCH.md Pattern 2).
PRECOMPUTED_DIR_NAME = ".precomputed"

#: Default video-only source map: directory name -> output key in the returned sample.
DEFAULT_DATA_SOURCES: dict[str, str] = {
    "latents": "latent_conditions",
    "conditions": "text_conditions",
}

#: The source dirs whose payloads are VIDEO VAE latents and therefore route through
#: ``_normalize_video_latents`` (the legacy patchified ``[seq_len,C]`` -> ``[C,F,H,W]`` einops
#: fixup). Gated as an explicit ALLOWLIST (Phase 9 a2v, GATE-SPEC rev 2 item 4) rather than the old
#: ``"latent" in dir_name`` SUBSTRING test: ``audio_latents`` CONTAINS "latent" but is NOT a video
#: latent — its bare tensor has no ``"latents"`` key, so the old substring branch would trip
#: ``_normalize_video_latents`` and raise. ``video_masks`` already avoided the substring; the a2v
#: ``audio_latents`` source is the one that forced the allowlist. Add a new video-latent source here
#: only if its payload is a video-VAE latent dict of the same ``{"latents": [C,F,H,W], …}`` shape.
#:
#: MiniMax-H3 (Phase 10, H3-03) registers FOUR sources under an ``h3_``-PREFIXED dir convention:
#: ``h3_latents`` / ``h3_conditions`` / ``h3_reference_latents`` / ``h3_audio_latents``. The prefix is
#: deliberate: it keeps the LTX path byte-identical and makes it impossible for an H3 cache to be
#: paired into an LTX run (the dir names simply never collide). Only the TWO video-latent ones are
#: allowlisted. H3 latents come from a DIFFERENT VAE than LTX's (``in_channels`` 24, not 128) and the
#: H3 pre-encode is signet-NATIVE — there is no upstream ``process_dataset`` for H3 — so the stored
#: dict shape is OUR choice, and this phase commits ``h3_latents`` / ``h3_reference_latents`` to the
#: SAME ``{"latents": [C,F,H,W], …}`` dict shape, which is exactly what makes the existing
#: normalization correct for them. ``h3_audio_latents`` is the substring trap all over again (it
#: CONTAINS "latent" but carries AUDIO-VAE latents with no ``"latents"`` key) and ``h3_conditions``
#: holds text embeddings — routing either through ``_normalize_video_latents`` would raise, so
#: NEITHER is allowlisted. Proof: ``tests/test_h3_precomputed_sources.py``.
_VIDEO_LATENT_SOURCE_DIRS: frozenset[str] = frozenset(
    {"latents", "reference_latents", "h3_latents", "h3_reference_latents"}
)


class PrecomputedDataset(Dataset):
    """Generic dataset reading precomputed ``.pt`` from multiple sources, paired by rel path.

    Args:
        data_root: Root directory containing preprocessed data. If it holds a
            ``.precomputed`` subdir, that subdir is used.
        data_sources: One of:
            - ``None`` -> defaults to the video-only ``{"latents": "latent_conditions",
              "conditions": "text_conditions"}`` map.
            - a ``list[str]`` of directory names (output key == directory name).
            - a ``dict[str, str]`` mapping directory name -> output key.

    Note:
        Latents are always returned non-patchified ``[C, F, H, W]``. The legacy patchified
        ``[seq_len, C]`` format is auto-converted (carried forward for back-compat).
    """

    def __init__(
        self,
        data_root: str,
        data_sources: dict[str, str] | list[str] | None = None,
    ) -> None:
        super().__init__()

        self.data_root = self._setup_data_root(data_root)
        self.data_sources = self._normalize_data_sources(data_sources)
        self.source_paths = self._setup_source_paths()
        self.sample_files = self._discover_samples()
        self._validate_setup()

    @staticmethod
    def _setup_data_root(data_root: str) -> Path:
        """Resolve the data root; descend into ``.precomputed`` if present."""
        root = Path(data_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {root}")
        if (root / PRECOMPUTED_DIR_NAME).exists():
            root = root / PRECOMPUTED_DIR_NAME
        return root

    @staticmethod
    def _normalize_data_sources(
        data_sources: dict[str, str] | list[str] | None,
    ) -> dict[str, str]:
        """Normalize the source spec to a ``{dir_name: output_key}`` dict (video-only default)."""
        if data_sources is None:
            return dict(DEFAULT_DATA_SOURCES)
        if isinstance(data_sources, list):
            return {source: source for source in data_sources}
        if isinstance(data_sources, dict):
            return data_sources.copy()
        raise TypeError(f"data_sources must be dict, list, or None, got {type(data_sources)}")

    def _setup_source_paths(self) -> dict[str, Path]:
        """Map each source directory name to its on-disk path; assert each exists."""
        source_paths: dict[str, Path] = {}
        for dir_name in self.data_sources:
            source_path = self.data_root / dir_name
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Required {dir_name} directory does not exist: {source_path}"
                )
            source_paths[dir_name] = source_path
        return source_paths

    def _discover_samples(self) -> dict[str, list[Path]]:
        """Index valid samples via a fast two-pass glob + set-membership pairing.

        Pass 1 globs every source's ``**/*.pt`` in parallel into full-path sets. Pass 2
        keeps only primary (latent) files that have a matching file in every other source.
        """
        if not self.data_sources:
            raise ValueError("No data sources configured")

        data_key = "latents" if "latents" in self.data_sources else next(iter(self.data_sources))
        data_path = self.source_paths[data_key]

        def _glob_source(dir_name: str) -> tuple[list[Path], set[str]]:
            paths = list(self.source_paths[dir_name].glob("**/*.pt"))
            return paths, {str(p) for p in paths}

        with ThreadPoolExecutor(max_workers=len(self.data_sources)) as executor:
            glob_results = dict(
                zip(
                    self.data_sources.keys(),
                    executor.map(_glob_source, self.data_sources.keys()),
                    strict=True,
                )
            )

        data_files, _ = glob_results[data_key]
        if not data_files:
            raise ValueError(f"No data files found in {data_path}")
        data_files.sort()

        for dir_name, (paths, _) in glob_results.items():
            logger.debug("Source %s: %d files", dir_name, len(paths))

        other_path_sets = {
            dir_name: path_set
            for dir_name, (_, path_set) in glob_results.items()
            if dir_name != data_key
        }

        sample_files: dict[str, list[Path]] = {
            output_key: [] for output_key in self.data_sources.values()
        }
        valid_count = 0
        for data_file in data_files:
            rel_path = data_file.relative_to(data_path)

            all_exist = True
            for dir_name, path_set in other_path_sets.items():
                expected = self._get_expected_file_path(dir_name, data_file, rel_path)
                if str(expected) not in path_set:
                    logger.debug(
                        "Skipping %s: no matching %s file at %s",
                        data_file.name,
                        dir_name,
                        expected,
                    )
                    all_exist = False
                    break

            if all_exist:
                self._fill_sample_data_files(data_file, rel_path, sample_files)
                valid_count += 1

        skipped = len(data_files) - valid_count
        if skipped > 0:
            logger.info(
                "Fast index: %d valid samples from %d total (%d skipped)",
                valid_count,
                len(data_files),
                skipped,
            )
        else:
            logger.debug("Fast index: %d valid samples from %d total", valid_count, len(data_files))

        return sample_files

    def _get_expected_file_path(self, dir_name: str, data_file: Path, rel_path: Path) -> Path:
        """Expected path of ``data_file`` under another source (with legacy condition naming)."""
        source_path = self.source_paths[dir_name]
        if dir_name == "conditions" and data_file.name.startswith("latent_"):
            return source_path / f"condition_{data_file.stem[7:]}.pt"
        return source_path / rel_path

    def _fill_sample_data_files(
        self, data_file: Path, rel_path: Path, sample_files: dict[str, list[Path]]
    ) -> None:
        """Record a validated sample's per-source relative paths under its output key."""
        for dir_name, output_key in self.data_sources.items():
            expected_path = self._get_expected_file_path(dir_name, data_file, rel_path)
            sample_files[output_key].append(
                expected_path.relative_to(self.source_paths[dir_name])
            )

    def _validate_setup(self) -> None:
        """Raise a clear error if no samples were paired, or counts mismatch across sources."""
        sample_counts = {key: len(files) for key, files in self.sample_files.items()}
        if not sample_counts or all(count == 0 for count in sample_counts.values()):
            raise ValueError(
                f"No valid samples found in {self.data_root} - all configured data sources "
                f"({list(self.data_sources)}) must have matching files "
                f"(per-source counts: {sample_counts})"
            )
        if len(set(sample_counts.values())) > 1:
            raise ValueError(f"Mismatched sample counts across sources: {sample_counts}")

    def __len__(self) -> int:
        first_key = next(iter(self.sample_files.keys()))
        return len(self.sample_files[first_key])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        result: dict[str, Tensor] = {}
        for dir_name, output_key in self.data_sources.items():
            source_path = self.source_paths[dir_name]
            file_rel_path = self.sample_files[output_key][index]
            file_path = source_path / file_rel_path
            try:
                data = torch.load(file_path, map_location="cpu", weights_only=True)
                # Route ONLY true video-VAE latent sources through the legacy patchified fixup.
                # GATE-SPEC rev 2 item 4: the old ``"latent" in dir_name`` SUBSTRING test tripped on
                # ``audio_latents`` (a2v) — an audio latent that has no ``"latents"`` key would then
                # raise inside ``_normalize_video_latents``. The explicit allowlist excludes it (and
                # any future non-video-latent source whose dir name happens to contain "latent").
                if dir_name in _VIDEO_LATENT_SOURCE_DIRS:
                    data = self._normalize_video_latents(data)
                result[output_key] = data
            except Exception as e:  # noqa: BLE001 — re-raise with the failing path for triage
                raise RuntimeError(f"Failed to load {output_key} from {file_path}: {e}") from e

        result["idx"] = index
        return result

    @staticmethod
    def _normalize_video_latents(data: dict) -> dict:
        """Convert legacy patchified ``[seq_len, C]`` latents to non-patchified ``[C, F, H, W]``."""
        latents = data["latents"]
        if latents.dim() == 2:
            from einops import rearrange  # lazy: legacy patchified path only (Phase-3 carry-forward)

            num_frames = data["num_frames"]
            height = data["height"]
            width = data["width"]
            latents = rearrange(
                latents,
                "(f h w) c -> c f h w",
                f=num_frames,
                h=height,
                w=width,
            )
            data = data.copy()
            data["latents"] = latents
        return data
