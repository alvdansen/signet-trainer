"""Throwaway CPU Modal probe (Phase 2 Task 4 preflight) — verify HF preconditions cheaply.

Checks, using the my-huggingface-secret token inside Modal:
  1. enochiatron-muxed access + structure (source clips for the D-01 fresh encode)
  2. Lightricks/LTX-2.3 gated-license acceptance (dev checkpoint)
  3. google/gemma-3-12b-it gated-license acceptance (text encoder)

Run: PYTHONIOENCODING=utf-8 python -m modal run scripts/_probe_hf_access.py
CPU only, no GPU, seconds of runtime. Delete after Phase 2.
"""

import modal

app = modal.App("signet-probe-hf-access")
image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub>=0.27")
hf_secret = modal.Secret.from_name("my-huggingface-secret")

# The private upstream source-clip dataset (enochiatron lineage) — point at your own HF dataset.
MUXED_REPO = "<hf-user>/enochiatron-muxed"


@app.function(image=image, secrets=[hf_secret], timeout=300)
def probe() -> None:
    import os
    from collections import Counter

    from huggingface_hub import auth_check, list_repo_files, whoami

    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        me = whoami(token=tok)
        print(f"[whoami] token belongs to: {me.get('name')} ({me.get('fullname','')})")
    except Exception as e:
        print(f"[whoami] FAILED: {type(e).__name__}: {e}")

    # 1. muxed source access + structure
    print(f"\n=== {MUXED_REPO} ===")
    try:
        files = list_repo_files(MUXED_REPO, repo_type="dataset", token=tok)
        print(f"  ACCESS OK — {len(files)} files")
        tops = Counter(f.split('/')[0] for f in files)
        for k, v in tops.items():
            print(f"    {k}/  ({v})")
        vids = [f for f in files if f.lower().endswith((".mp4", ".mov", ".mkv", ".webm"))]
        print(f"  {len(vids)} video files; first 8:")
        for f in vids[:8]:
            print("    ", f)
    except Exception as e:
        print(f"  NO ACCESS: {type(e).__name__}: {str(e)[:160]}")

    # 2 + 3. gated license acceptance (auth_check raises GatedRepoError if not accepted)
    for repo, rtype in [("Lightricks/LTX-2.3", "model"), ("google/gemma-3-12b-it", "model")]:
        print(f"\n=== {repo} ===")
        try:
            auth_check(repo, repo_type=rtype, token=tok)
            print("  LICENSE OK (access granted)")
        except Exception as e:
            print(f"  BLOCKED: {type(e).__name__}: {str(e)[:160]}")


@app.local_entrypoint()
def main() -> None:
    probe.remote()
