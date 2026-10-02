#!/usr/bin/env python
"""Upload the converted Parquet files and dataset card to the Hugging Face Hub.

Needs HF_TOKEN in the environment (a token with write access).

Usage:
    python scripts/publish_hf.py --repo <user>/qwen_robotics_open_dataset --data data/hf --card dataset_card.md
"""
import argparse
import os

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", default="data/hf")
    ap.add_argument("--card", default="dataset_card.md")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=args.data, path_in_repo="data",
                      commit_message="Add scenario parquet files", delete_patterns="*.parquet")
    if os.path.isdir(args.figures):
        api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=args.figures, path_in_repo="figures",
                          commit_message="Add figures")
    api.upload_file(repo_id=args.repo, repo_type="dataset", path_or_fileobj=args.card, path_in_repo="README.md",
                    commit_message="Update dataset card")
    print(f"https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
