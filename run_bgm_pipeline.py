#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot pipeline:
  1) run generate_bgm_package.py
  2) run upload_youtube.py

Example:
  python3 run_bgm_pipeline.py \
    --client_secrets ./client_secret.json \
    --privacy public \
    --out_dir ./output \
    -- --thumb_text --make_video --clock
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def run_cmd(cmd: List[str]) -> None:
    print("\n[RUN]", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode != 0:
        raise SystemExit(f"[ERROR] Command failed (code={p.returncode})")


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"[ERROR] {label} not found: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate BGM package then upload to YouTube in a single run."
    )

    # scripts
    ap.add_argument("--generate_script", default="./generate_bgm_package.py",
                    help="path to generate_bgm_package.py")
    ap.add_argument("--upload_script", default="./upload_youtube.py",
                    help="path to upload_youtube.py")

    # shared
    ap.add_argument("--out_dir", default="./output", help="output directory (same as generate script)")

    # upload options (minimum)
    ap.add_argument("--client_secrets", required=True, help="client_secret.json path")
    ap.add_argument("--token", default="./.yt_token/token.json", help="token json path")
    ap.add_argument("--privacy", default="private", choices=["private", "unlisted", "public"])
    ap.add_argument("--category_id", default="10", help="10=Music, 27=Education, ...")
    ap.add_argument("--publish_at", default=None, help="RFC3339. e.g. 2026-01-08T09:00:00+09:00")
    ap.add_argument("--made_for_kids", action="store_true")

    # tags (upload_youtube.py 側の default が 'DEFAULT_TAGS' 文字列になっている場合の保険)
    ap.add_argument("--tags", default=None,
                    help="comma separated tags (optional). If omitted, upload_youtube.py's default is used.")

    # behavior
    ap.add_argument("--skip_generate", action="store_true",
                    help="skip generation step and only upload existing files in out_dir")
    ap.add_argument("--dry_run", action="store_true", help="print commands only")

    # pass-through args to generator after `--`
    # e.g. -- --thumb_text --make_video --clock
    args, gen_args = ap.parse_known_args()

    out_dir = Path(args.out_dir)
    gen_script = Path(args.generate_script)
    up_script = Path(args.upload_script)

    if not gen_script.exists():
        raise SystemExit(f"[ERROR] generate_script not found: {gen_script}")
    if not up_script.exists():
        raise SystemExit(f"[ERROR] upload_script not found: {up_script}")

    # 1) generate
    if not args.skip_generate:
        cmd_gen = [sys.executable, str(gen_script), "--out_dir", str(out_dir)] + gen_args
        if args.dry_run:
            print("[DRY RUN]", " ".join(cmd_gen))
        else:
            run_cmd(cmd_gen)

    # 2) check outputs
    video = out_dir / "video.mp4"
    thumb = out_dir / "thumbnail.jpg"
    title_file = out_dir / "title.txt"
    desc_file = out_dir / "description.txt"

    ensure_exists(video, "video")
    ensure_exists(thumb, "thumbnail")
    ensure_exists(title_file, "title_file")
    ensure_exists(desc_file, "description_file")

    # 3) upload
    cmd_up = [
        sys.executable, str(up_script),
        "--client_secrets", str(Path(args.client_secrets)),
        "--token", str(Path(args.token)),
        "--video", str(video),
        "--thumb", str(thumb),
        "--title_file", str(title_file),
        "--description_file", str(desc_file),
        "--privacy", args.privacy,
        "--category_id", args.category_id,
    ]
    if args.publish_at:
        cmd_up += ["--publish_at", args.publish_at]
    if args.made_for_kids:
        cmd_up += ["--made_for_kids"]
    if args.tags:
        cmd_up += ["--tags", args.tags]

    if args.dry_run:
        print("[DRY RUN]", " ".join(cmd_up))
    else:
        run_cmd(cmd_up)

    print("\n[DONE] Generated + Uploaded successfully.")


if __name__ == "__main__":
    main()
