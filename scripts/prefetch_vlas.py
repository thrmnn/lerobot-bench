"""Pre-download all 5 VLA checkpoints (and shared paligemma-3b base) to
HF cache. No model instantiation, no GPU. IO-only.

Logs progress to stdout with sizes. Call before Phase 3 to decouple
"weights downloaded" from "inference works".
"""

from __future__ import annotations

import logging
import shutil
import sys
import time

from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prefetch")

# (repo_id, revision_sha) — kept in sync with configs/policies.yaml
TARGETS = [
    ("lerobot/smolvla_libero", "31d453f7edd78c839a8bbc39744a292686daf0de"),
    ("lerobot/xvla-libero", "12e8783e996944f5c97e490d37d4c145484ed70a"),
    ("lerobot/pi0fast-libero", "840f4b503f4c09110421c33c810a85b6684fd658"),
    ("lerobot/pi0_libero_finetuned_v044", "45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f"),
    ("lerobot/pi05_libero_finetuned_v044", "dbf8a3f794a9c4297b44f40b752712f50073d945"),
]


def disk_free_gb() -> float:
    return shutil.disk_usage("/home").free / (1024**3)


def main() -> int:
    logger.info("disk free at start: %.1f GB", disk_free_gb())
    rc = 0
    for repo_id, revision in TARGETS:
        t0 = time.time()
        free_before = disk_free_gb()
        logger.info("downloading %s @ %s ...", repo_id, revision[:8])
        try:
            local_dir = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                # Don't symlink — we want a real cache entry that survives
                # multiple repos sharing the same paligemma-3b base.
                local_dir=None,
            )
            free_after = disk_free_gb()
            took_gb = max(0.0, free_before - free_after)
            took_s = time.time() - t0
            logger.info(
                "  ok %s -> %s (took %.1fs, +%.2f GB on disk)",
                repo_id,
                local_dir,
                took_s,
                took_gb,
            )
        except Exception as exc:
            logger.error("  FAILED %s: %s", repo_id, exc)
            rc = 1
    logger.info("disk free at end: %.1f GB", disk_free_gb())
    return rc


if __name__ == "__main__":
    sys.exit(main())
