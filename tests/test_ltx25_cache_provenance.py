"""data.cache_provenance (issue #53 Stage 1) — PROVENANCE.json write/read/assert, CPU only.

Covers the two refuter MUST-FIX findings directly:
  * a generation-only comparison cannot catch a Gemma-ROOT swap under the SAME generation
    (assert_cache_provenance now takes configured_gemma_root too);
  * the mixed-cache case: a non-empty output_dir lacking matching provenance must be refused
    UNLESS overwrite=True forces a full re-encode (assert_output_dir_ready_for_ltx25_encode).
"""

from __future__ import annotations

import pytest

from signet_trainer.data.cache_provenance import (
    assert_cache_provenance,
    assert_output_dir_ready_for_ltx25_encode,
    read_provenance,
    write_provenance,
)

# ======================================================================================
# write_provenance / read_provenance roundtrip.
# ======================================================================================


def test_write_then_read_roundtrip(tmp_path) -> None:
    write_provenance(str(tmp_path), ltx_generation="2.5", gemma_root="gemma-4-12b-it")
    recorded = read_provenance(str(tmp_path))
    assert recorded == {"ltx_generation": "2.5", "gemma_root": "gemma-4-12b-it"}


def test_read_provenance_missing_file_returns_none(tmp_path) -> None:
    assert read_provenance(str(tmp_path)) is None


def test_read_provenance_unparseable_file_returns_none(tmp_path) -> None:
    (tmp_path / "PROVENANCE.json").write_text("not json {{{", encoding="utf-8")
    assert read_provenance(str(tmp_path)) is None


# ======================================================================================
# assert_cache_provenance — generation AND Gemma-root comparison.
# ======================================================================================


def test_write_then_assert_roundtrip_passes(tmp_path) -> None:
    write_provenance(str(tmp_path), ltx_generation="2.5", gemma_root="gemma-4-12b-it")
    assert_cache_provenance(
        str(tmp_path), configured_generation="2.5", configured_gemma_root="gemma-4-12b-it"
    )


def test_missing_provenance_refused_only_for_25(tmp_path) -> None:
    """An output_dir with no PROVENANCE.json: passes silently for a '2.3'-configured run, raises
    for a '2.5'-configured run."""
    assert_cache_provenance(
        str(tmp_path), configured_generation="2.3", configured_gemma_root="gemma-3-12b-it"
    )
    with pytest.raises(ValueError, match="LEGACY 2.3"):
        assert_cache_provenance(
            str(tmp_path), configured_generation="2.5", configured_gemma_root="gemma-4-12b-it"
        )


def test_generation_mismatch_refused_even_though_both_are_present(tmp_path) -> None:
    """A cache provenanced for '2.3' is never accepted for a '2.5' run — a 3-argument generation
    mismatch is refused even though provenance IS present (not merely absent)."""
    write_provenance(str(tmp_path), ltx_generation="2.3", gemma_root="gemma-3-12b-it")
    with pytest.raises(ValueError, match="ltx_generation"):
        assert_cache_provenance(
            str(tmp_path), configured_generation="2.5", configured_gemma_root="gemma-4-12b-it"
        )


def test_gemma_root_mismatch_refused_under_the_same_generation(tmp_path) -> None:
    """Refuter MUST-FIX: a SAME-generation Gemma-root swap (e.g. re-pointing text_encoder_id at a
    different -it/-qat variant) must be caught — a generation-only check cannot see this."""
    write_provenance(str(tmp_path), ltx_generation="2.5", gemma_root="gemma-4-12b-it")
    with pytest.raises(ValueError, match="gemma_root"):
        assert_cache_provenance(
            str(tmp_path),
            configured_generation="2.5",
            configured_gemma_root="gemma-4-12b-qat-q4_0",
        )


def test_gemma_root_is_not_compared_under_generation_23() -> None:
    """Byte-identity: a '2.3'-configured run never reads/compares gemma_root at all (no 2.3
    workflow writes provenance, so this path is unreachable for it in practice — but the
    function itself must not raise if it somehow were called this way)."""
    assert_cache_provenance(
        "/does/not/exist", configured_generation="2.3", configured_gemma_root="anything"
    )


# ======================================================================================
# assert_output_dir_ready_for_ltx25_encode — the mixed-cache-refusal gate.
# ======================================================================================


def test_fresh_nonexistent_dir_is_always_ready(tmp_path) -> None:
    fresh = tmp_path / "brand_new"
    assert_output_dir_ready_for_ltx25_encode(str(fresh), gemma_root="gemma-4-12b-it", overwrite=False)


def test_empty_existing_dir_is_ready(tmp_path) -> None:
    assert_output_dir_ready_for_ltx25_encode(str(tmp_path), gemma_root="gemma-4-12b-it", overwrite=False)


def test_nonempty_dir_with_no_provenance_is_refused_without_overwrite(tmp_path) -> None:
    (tmp_path / "conditions").mkdir()
    (tmp_path / "conditions" / "stale.pt").write_bytes(b"stale-2.3-gemma3-cache")

    with pytest.raises(ValueError, match="mixed"):
        assert_output_dir_ready_for_ltx25_encode(
            str(tmp_path), gemma_root="gemma-4-12b-it", overwrite=False
        )


def test_nonempty_dir_with_no_provenance_is_allowed_with_overwrite_true(tmp_path) -> None:
    """overwrite=True forces a FULL re-encode -- the mixed-cache risk cannot occur because every
    sample gets re-produced under the new generation/Gemma root."""
    (tmp_path / "conditions").mkdir()
    (tmp_path / "conditions" / "stale.pt").write_bytes(b"stale-2.3-gemma3-cache")

    assert_output_dir_ready_for_ltx25_encode(
        str(tmp_path), gemma_root="gemma-4-12b-it", overwrite=True
    )


def test_nonempty_dir_with_matching_provenance_is_allowed_without_overwrite(tmp_path) -> None:
    """An incremental ltx25-on-ltx25 re-run against its OWN prior output (matching generation AND
    gemma_root) is fine without overwrite."""
    write_provenance(str(tmp_path), ltx_generation="2.5", gemma_root="gemma-4-12b-it")
    (tmp_path / "conditions").mkdir()
    (tmp_path / "conditions" / "existing.pt").write_bytes(b"already-2.5-gemma4")

    assert_output_dir_ready_for_ltx25_encode(
        str(tmp_path), gemma_root="gemma-4-12b-it", overwrite=False
    )


def test_nonempty_dir_with_provenance_for_a_different_gemma_root_is_refused(tmp_path) -> None:
    """The mixed-cache case extends to a root swap under the SAME generation, not just a
    generation swap — matches the same refuter finding assert_cache_provenance closes."""
    write_provenance(str(tmp_path), ltx_generation="2.5", gemma_root="gemma-4-12b-it")
    (tmp_path / "conditions").mkdir()
    (tmp_path / "conditions" / "existing.pt").write_bytes(b"gemma4-it-cache")

    with pytest.raises(ValueError, match="mixed"):
        assert_output_dir_ready_for_ltx25_encode(
            str(tmp_path), gemma_root="gemma-4-12b-qat-q4_0", overwrite=False
        )


def test_nonempty_dir_with_23_provenance_is_refused_for_a_25_encode(tmp_path) -> None:
    write_provenance(str(tmp_path), ltx_generation="2.3", gemma_root="gemma-3-12b-it")
    (tmp_path / "conditions").mkdir()
    (tmp_path / "conditions" / "existing.pt").write_bytes(b"gemma3-cache")

    with pytest.raises(ValueError, match="mixed"):
        assert_output_dir_ready_for_ltx25_encode(
            str(tmp_path), gemma_root="gemma-4-12b-it", overwrite=False
        )
