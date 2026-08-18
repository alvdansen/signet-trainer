"""BK-01 — BackupConfig is a config-first, DEFAULT-OFF, mirror-all, private-HF contract.

Config-first / D-NOHARDCODE (D-BK-1/2/5): the auto-backup surface is a documented, defaulted
Pydantic block wired into SignetConfig default-off, so every existing YAML loads byte-identically.
Its cross-field validator is SYMMETRIC across all three destinations (each rejects every stray
field a mode wouldn't consume) and fail-fasts an ENABLED destination='local' OR 'cloud' block AT
CONFIG LOAD (never deep in a Modal function) — 'local' because a Modal container's filesystem is
not durable (issue #23 finding 1: backup_sync never commits a Volume for it), 'cloud' because it is
not yet implemented.

Asserts, CPU-only (no modal / torch / GPU import):
  * a backup-unaware SignetConfig loads with enabled=False, what='all', destination='hf', private=True;
  * enabled + hf + no repo_id raises; enabled + hf + stray path raises;
  * enabled + local (otherwise clean) raises "not durable on Modal" (issue #23 finding 1);
  * enabled + local + stray repo_id/cloud_secret_name raises (symmetric stray-field rejection);
  * disabled + local loads clean (inert block, byte-identical to today);
  * what != 'final+intervals' + interval set raises;
  * a typo key inside the backup block raises (extra='forbid');
  * enabled + cloud (otherwise clean) raises "not yet implemented";
  * enabled + cloud + stray repo_id/path raises (symmetric stray-field rejection);
  * disabled + cloud loads clean (inert block).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from signet_trainer.config.schema import BackupConfig, SignetConfig


def _minimal_payload(backup: dict | None = None) -> dict:
    # A minimal SignetConfig; the backup block is optional (default-off) — matching the composing schema.
    payload = {
        "training_dims": [768, 512, 49],
        "data": {"preprocessed_data_root": "/data/preprocessed", "batch_size": 1},
        "training": {"max_steps": 100},
    }
    if backup is not None:
        payload["backup"] = backup
    return payload


# ---- (a) default-off byte-identical: a backup-unaware config loads with the house defaults ----


def test_backup_unaware_config_loads_with_house_defaults():
    cfg = SignetConfig.model_validate(_minimal_payload())
    assert cfg.backup.enabled is False  # off-default = byte-identical current behavior
    assert cfg.backup.what == "all"  # D-BK-2 house default (mirror-all, never final-only)
    assert cfg.backup.destination == "hf"  # D-BK-5 durable off-Volume default
    assert cfg.backup.private is True  # D-BK-5 private-by-default
    assert cfg.backup.repo_id is None
    assert cfg.backup.path is None
    assert cfg.backup.cloud_secret_name is None
    assert cfg.backup.interval is None


def test_backup_fields_are_documented():
    # D-NOHARDCODE: every field is a config-visible documented Field, not an anonymous literal.
    for name in (
        "enabled",
        "destination",
        "repo_id",
        "path",
        "cloud_secret_name",
        "private",
        "what",
        "interval",
    ):
        assert BackupConfig.model_fields[name].description


# ---- (b)/(c) symmetric hf validator: repo_id required; stray path rejected ----


def test_enabled_hf_without_repo_id_raises():
    with pytest.raises(ValidationError, match="requires backup.repo_id"):
        BackupConfig(enabled=True, destination="hf", repo_id=None)


def test_enabled_hf_with_stray_path_raises():
    with pytest.raises(ValidationError, match="path is set while destination='hf'"):
        BackupConfig(enabled=True, destination="hf", repo_id="owner/repo", path="/tmp/backups")


def test_enabled_hf_with_stray_cloud_secret_raises():
    with pytest.raises(ValidationError, match="cloud_secret_name is set while destination='hf'"):
        BackupConfig(
            enabled=True, destination="hf", repo_id="owner/repo", cloud_secret_name="my-cloud"
        )


def test_enabled_hf_with_repo_id_loads():
    b = BackupConfig(enabled=True, destination="hf", repo_id="owner/repo")
    assert b.destination == "hf"
    assert b.repo_id == "owner/repo"
    assert b.private is True


# ---- (d) local fail-fast: not durable on Modal (issue #23 finding 1), plus symmetric stray-field
#          rejection and inert-when-disabled (byte-identical to today) ----


def test_enabled_local_clean_block_raises_not_durable_on_modal():
    # issue #23 finding 1: destination='local' writes into the Modal container's own ephemeral
    # filesystem and no code ever commits it to a Volume, so the copy would report success and then
    # vanish. Fail fast at config load, mirroring the cloud precedent exactly, rather than let a
    # non-durable "backup" run to a false "copied N dir(s)" success line.
    with pytest.raises(ValidationError, match="not durable on Modal"):
        BackupConfig(enabled=True, destination="local", path="/backups/run1")


def test_enabled_local_without_path_also_raises_not_durable():
    # 'local' fails fast unconditionally when enabled — even with path=None, the "not durable"
    # message fires (not a stray/missing-field message), because the destination itself is refused.
    with pytest.raises(ValidationError, match="not durable on Modal"):
        BackupConfig(enabled=True, destination="local", path=None)


def test_enabled_local_with_stray_repo_id_raises_symmetric():
    with pytest.raises(ValidationError, match="repo_id is set while destination='local'"):
        BackupConfig(enabled=True, destination="local", path="/backups", repo_id="owner/repo")


def test_enabled_local_with_stray_cloud_secret_raises_symmetric():
    with pytest.raises(ValidationError, match="cloud_secret_name is set while destination='local'"):
        BackupConfig(enabled=True, destination="local", cloud_secret_name="my-cloud-secret")


def test_disabled_local_loads_clean():
    # Byte-identical-when-off contract (schema.py's enabled=False precedent): a disabled 'local'
    # block — even one that carries a path, as an operator's pre-existing YAML might — must NOT
    # raise. Only an ENABLED local block is refused.
    b = BackupConfig(enabled=False, destination="local", path="/backups/run1")
    assert b.enabled is False
    assert b.destination == "local"
    assert b.path == "/backups/run1"
    cfg = SignetConfig.model_validate(
        _minimal_payload({"enabled": False, "destination": "local", "path": "/backups/run1"})
    )
    assert cfg.backup.enabled is False
    assert cfg.backup.destination == "local"


# ---- (e) interval only valid with what='final+intervals' ----


def test_interval_without_final_intervals_raises():
    with pytest.raises(ValidationError, match="interval is set"):
        BackupConfig(what="all", interval=1000)


def test_interval_with_final_intervals_loads():
    b = BackupConfig(what="final+intervals", interval=1000)
    assert b.what == "final+intervals"
    assert b.interval == 1000


# ---- (f) extra='forbid' rejects a typo key inside the backup block ----


def test_typo_key_in_backup_block_raises():
    with pytest.raises(ValidationError):
        SignetConfig.model_validate(_minimal_payload({"enabled": False, "destinaton": "hf"}))


# ---- (g)/(h) cloud fail-fast: not-yet-implemented, plus symmetric stray-field rejection ----


def test_enabled_cloud_clean_block_raises_not_yet_implemented():
    with pytest.raises(ValidationError, match="not yet implemented"):
        BackupConfig(enabled=True, destination="cloud", cloud_secret_name="my-cloud-secret")


def test_enabled_cloud_with_stray_repo_id_raises_symmetric():
    with pytest.raises(ValidationError, match="repo_id is set while destination='cloud'"):
        BackupConfig(enabled=True, destination="cloud", repo_id="owner/repo")


def test_enabled_cloud_with_stray_path_raises_symmetric():
    with pytest.raises(ValidationError, match="path is set while destination='cloud'"):
        BackupConfig(enabled=True, destination="cloud", path="/backups")


# ---- (i) disabled cloud block is inert and loads clean ----


def test_disabled_cloud_loads_clean():
    b = BackupConfig(enabled=False, destination="cloud")
    assert b.enabled is False
    assert b.destination == "cloud"
    # And through the composing config (default-off, cloud selected but inert).
    cfg = SignetConfig.model_validate(_minimal_payload({"enabled": False, "destination": "cloud"}))
    assert cfg.backup.enabled is False
    assert cfg.backup.destination == "cloud"
