"""Run a trained navigation policy on your own images: images in, path out.

    python -m vla.infer --hub-repo Jinyan0924/qwen_robotics_nav_policy --checkpoint e1_all_sqrt/best \
        --images t-5.jpg t-4.jpg t-3.jpg t-2.jpg t-1.jpg now.jpg --goal 4.0 -1.0 --plot path.png

Images are the robot's front camera, oldest first, one per `past_dt_s` seconds (1 s for run e1); the last is
the current view.  The goal is in metres in the robot frame: robot at (0, 0), x forward, y left.  Prints the
waypoints (x, y), `spacing_m` apart along the path (8 x 0.25 m for run e1).  From Python:

    from vla.pipeline import NavigationPipeline
    pipe = NavigationPipeline.from_hub("Jinyan0924/qwen_robotics_nav_policy", "e1_all_sqrt/best")
    path = pipe.predict_path(images, goal=(4.0, -1.0))          # (8, 2) metres

To score a checkpoint on the evaluation suite use `python -m vla.predict_suite` instead.
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla.pipeline import NavigationPipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="local folder, or a folder of --hub-repo")
    ap.add_argument("--hub-repo", default="")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--images", nargs="+", required=True, help="image files, oldest first; the last is now")
    ap.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"))
    ap.add_argument("--instruction", default="")
    ap.add_argument("--past", default="", help='JSON list of past (x, y) in the current robot frame, oldest first')
    ap.add_argument("--plot", default="", help="write the current image with the bird's-eye path to this file")
    args = ap.parse_args()
    pipe = (NavigationPipeline.from_hub(args.hub_repo, args.checkpoint, args.revision) if args.hub_repo
            else NavigationPipeline.from_pretrained(args.checkpoint))
    images = [Image.open(p) for p in args.images]
    path = pipe.predict_path(images, goal=args.goal, instruction=args.instruction or None,
                             past_xy=json.loads(args.past) if args.past else None)
    print(json.dumps([[round(float(x), 3), round(float(y), 3)] for x, y in path]))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw=dict(width_ratios=[1.6, 1]))
        ax[0].imshow(images[-1])
        ax[0].axis("off")
        ax[1].plot(-path[:, 1], path[:, 0], "o-", c="tab:green", label="predicted path")
        ax[1].plot(0, 0, "s", c="k", label="robot")
        if args.goal:
            ax[1].plot(-args.goal[1], args.goal[0], "*", c="gold", mec="k", ms=16, label="goal")
        ax[1].set_aspect("equal")
        ax[1].grid(alpha=.3)
        ax[1].set_xlabel("right (m)")
        ax[1].set_ylabel("forward (m)")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=80)


if __name__ == "__main__":
    main()
