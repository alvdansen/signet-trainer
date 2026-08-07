"""YAML -> SignetConfig loader (CONF-01, D-05). The Phase-8-harness ``load_config(yaml)`` seam.

A clean, importable top-level function — NOT inline, NOT a hand-rolled ``setattr`` parser
(CONF-01 forbids a hand-rolled parser). Uses ``yaml.safe_load`` (never ``yaml.load`` /
``Loader=FullLoader``) so a malicious YAML cannot execute arbitrary Python (T-01-CF1), then
``SignetConfig.model_validate`` (Pydantic) which fires all CONF-02 fail-fast validators.

NO ``modal`` / ``ltx_core`` import anywhere in this module (Anti-Pattern 6).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from signet_trainer.config.schema import SignetConfig


def load_config(path: str | Path) -> SignetConfig:
    """Load a YAML run config into a validated ``SignetConfig``.

    Raises ``pydantic.ValidationError`` (fail-fast, before any GPU) if any LTX constraint
    is violated — e.g. a bad frame count exits HERE with a clear message.
    """
    text = Path(path).read_text(encoding="utf-8")
    return load_config_from_text(text)


def load_config_from_text(text: str) -> SignetConfig:
    """Validate a YAML config from its RAW TEXT (same validators as ``load_config``).

    Used to ship a run config to a remote Modal container BY VALUE: the container re-parses +
    re-validates the exact text the entrypoint validated locally, so it never needs the config
    file on its own filesystem (the ``configs/`` dir is not shipped into the image). Preserves the
    fail-fast CONF-02 validation on the remote side before any sustained GPU spend.
    """
    data = yaml.safe_load(text)
    if data is None:
        data = {}
    return SignetConfig.model_validate(data)
