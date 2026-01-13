#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Open AIのAPIキーの発行に金がかかるので、一旦保留
import argparse
import base64
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # pip install pyyaml
from openai import OpenAI  # pip install openai


@dataclass
class PromptItem:
    name: str
    text: str


def date_seed_int() -> int:
    return int(datetime.now().strftime("%Y%m%d"))


def load_prompts_yaml(path: Path) -> (str, List[PromptItem]):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    common_suffix = (data.get("common_suffix") or "").strip()
    items = []
    for p in data.get("prompts", []):
        items.append(PromptItem(name=str(p["name"]), text=str(p["text"])))
    if not items:
        raise RuntimeError(f"prompts が空です: {path}")
    return common_suffix, items


def safe_filename(s: str) -> str:
    # ファイル名に使えない文字を軽く潰す
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in s).strip("_")


def save_b64_image(b64: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = base64.b64decode(b64)
    out_path.write_bytes(raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts_yaml", required=True, help="prompts.yaml のパス")
    ap.add_argument("--out_dir", required=True, help="保存先ディレクトリ（例: ./images）")
    ap.add_argument("--each", type=int, default=1, help="各プロンプトから生成する枚数")
    ap.add_argument("--size", default="1792x1024", help="例: 1792x1024 / 1024x1024")
    ap.add_argument("--model", default="gpt-image-1")
    ap.add_argument("--seed", type=int, default=None, help="未指定なら日付ベース")
    ap.add_argument("--shuffle", action="store_true", help="プロンプト順をシャッフル")
    ap.add_argument("--prefix", default="bgm", help="出力ファイル名の接頭辞")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else date_seed_int()
    rng = random.Random(seed)

    prompts_yaml = Path(args.prompts_yaml)
    out_dir = Path(args.out_dir)

    common_suffix, prompts = load_prompts_yaml(prompts_yaml)
    if args.shuffle:
        rng.shuffle(prompts)

    client = OpenAI()  # OPENAI_API_KEY を環境変数で設定しておく

    manifest: List[Dict[str, Any]] = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for item in prompts:
        for j in range(args.each):
            full_prompt = item.text.strip()
            if common_suffix:
                full_prompt = f"{full_prompt}, {common_suffix}"

            # 生成リクエスト（b64で受け取って保存）
            # 公式ガイドに沿って images.generate を使用 :contentReference[oaicite:3]{index=3}
            result = client.images.generate(
                model=args.model,
                prompt=full_prompt,
                size=args.size,
            )

            b64 = result.data[0].b64_json
            name = f"{args.prefix}_{ts}_{safe_filename(item.name)}_{j+1:02d}.png"
            out_path = out_dir / name
            save_b64_image(b64, out_path)

            manifest.append(
                {
                    "name": item.name,
                    "index": j + 1,
                    "prompt": full_prompt,
                    "size": args.size,
                    "path": str(out_path),
                    "seed": seed,
                    "created_at": ts,
                }
            )
            print("Saved:", out_path)

    (out_dir / f"manifest_{ts}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("DONE")
    print("seed:", seed)
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
