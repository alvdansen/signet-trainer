"""``signet_trainer.prep`` — the inpaint data-prep package (D-13).

The shared, importable, spec-aware core the four r1 scripts are formalized into (spec loader,
stem-resolution dedup, dims preflight, chroma rebuild, SAM propagate, staging, masking app). Kept
deliberately thin at the package root — NO heavy re-exports here — so downstream plans add sibling
modules (chroma / propagate / stage / app) without editing this file, and so ``import
signet_trainer.prep`` never drags torch/cv2/modal. Import submodules directly
(``from signet_trainer.prep.spec import load_mask_spec``).
"""
