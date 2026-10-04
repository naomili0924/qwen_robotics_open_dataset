"""Keep a run's latest checkpoint on the Hugging Face Hub so that training survives the machine.

With cfg.hub_repo set, every checkpoint is mirrored to <hub_repo>/<run name>/last together with the
training log, and `--resume auto` continues from the newest copy: the local <run>/last if it exists,
otherwise the one on the Hub.  Needs HF_TOKEN.  Upload failures are reported, never fatal.
"""
import os
import shutil
from pathlib import Path


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN"))


def push_checkpoint(run, hub_repo, private=True):
    """Upload <run>/last, log.jsonl and config.json to <hub_repo>/<run name>/."""
    run = Path(run)
    try:
        api = _api()
        api.create_repo(hub_repo, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(repo_id=hub_repo, repo_type="model", folder_path=str(run / "last"),
                          path_in_repo=f"{run.name}/last", commit_message=f"{run.name}: checkpoint")
        for name in ("log.jsonl", "config.json"):
            if (run / name).exists():
                api.upload_file(repo_id=hub_repo, repo_type="model", path_or_fileobj=str(run / name),
                                path_in_repo=f"{run.name}/{name}", commit_message=f"{run.name}: {name}")
        return True
    except Exception as e:  # a network problem must not stop training
        print(f"hub upload failed ({hub_repo}): {e!r}"[:300], flush=True)
        return False


def resolve_resume(cfg):
    """Checkpoint directory for cfg.resume ("" none, "auto", or a path); fetches from the Hub if needed."""
    if cfg.resume != "auto":
        return cfg.resume
    run = Path(cfg.run)
    if (run / "last" / "trainer.pt").exists():
        return str(run / "last")
    if cfg.hub_repo:
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(cfg.hub_repo, repo_type="model", token=os.environ.get("HF_TOKEN"), local_dir=str(run.parent),
                              allow_patterns=[f"{run.name}/last/*", f"{run.name}/last/**", f"{run.name}/log.jsonl"])
            if (run / "last" / "trainer.pt").exists():
                print(f"fetched {run.name}/last from {cfg.hub_repo}", flush=True)
                return str(run / "last")
        except Exception as e:
            print(f"no checkpoint fetched from {cfg.hub_repo}: {e!r}"[:300], flush=True)
    return ""


def save_last(run, name, hub_repo):
    """After <run>/<name> was written: make it <run>/last as well and mirror it to the Hub."""
    run = Path(run)
    if name != "last":
        tmp = run / "last.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(run / name, tmp)
        shutil.rmtree(run / "last", ignore_errors=True)
        tmp.rename(run / "last")
    if hub_repo:
        push_checkpoint(run, hub_repo)
