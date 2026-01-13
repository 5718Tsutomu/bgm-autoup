#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Example:
#   python3 generate_bgm_package.py \
#  --thumb_text \
#  --make_video \
#  --clock \
#  --out_dir ./output

import argparse
import random
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict

from PIL import Image, ImageOps, ImageDraw, ImageFont

# =========================
# Defaults (user specified)
# =========================
DEFAULT_BGM_DIR = str(Path(__file__).resolve().parent / "bgms")
DEFAULT_IMG_DIR = str(Path(__file__).resolve().parent / "images")

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# default candidates file path (same directory as this script)
DEFAULT_THUMB_TEXT_FILE = Path(__file__).with_name("thumbnail_text_candidates.txt")

# =========================
# Title map (chosen_text -> title)
# ※ thumb text candidates と同じキーを用意してください
# =========================
TITLE_MAP: Dict[str, str] = {
    "2H FOCUS BGM": "2 Hours Focus BGM | Lyric-Free Background Music for Study & Work",
    "STUDY / WORK": "2 Hours Study / Work | Lyric-Free Background Music (No Lyrics)",
    "DEEP WORK": "2 Hours Deep Work | Lyric-Free Background Music for Focus (No Lyrics)",
    "NO LYRICS": "2 Hours No Lyrics | Background Music for Focus, Study & Work",
    "LOFI / AMBIENT": "2 Hours Lofi / Ambient | Calm Background Music (No Lyrics)",
    "CALM & PRODUCTIVE": "2 Hours Calm & Productive | Background Music for Deep Focus",
    "CHILL STUDY MIX": "2 Hours Chill Study Mix | Lyric-Free Background Music",
    "CONCENTRATION MODE": "2 Hours Concentration Mode | Background Music for Study & Work",
    "SOFT BACKGROUND": "2 Hours Soft Background | Minimal Music for Focus (No Lyrics)",
    "QUIET HOURS": "2 Hours Quiet Hours | Calm Background Music for Work & Study",
    "WORK SESSION": "2 Hours Work Session | Lyric-Free Background Music for Productivity",
    "RELAX / STUDY": "2 Hours Relax / Study | Calm Background Music (No Lyrics)",
    "BEATS TO STUDY TO": "2 Hours Beats to Study To | Lyric-Free Background Music",
    "ULTRA FOCUS": "2 Hours Ultra Focus | Lyric-Free Background Music for Study & Work",
    "FLOW STATE": "2 Hours Flow State | Deep Focus Background Music (No Lyrics)",
    "CODING MODE": "2 Hours Coding Mode | Background Music for Programming (No Lyrics)",
    "READING SESSION": "2 Hours Reading Session | Calm Background Music (No Lyrics)",
    "DEEP CONCENTRATION": "2 Hours Deep Concentration | Lyric-Free Background Music",
    "ZERO DISTRACTIONS": "2 Hours Zero Distractions | Minimal Background Music (No Lyrics)",
    "STUDY SPRINT": "2 Hours Study Sprint | Background Music for Focus (No Lyrics)",
}

# =========================
# Description header (English only)
# =========================
DESCRIPTION_HEADER = """2 hours of lyric-free background music for deep focus, studying, and work.

🎧 Recommended: headphones / moderate volume.
✅ All tracks in this video are selected from the YouTube Audio Library.
ℹ️ If any track requires attribution (Creative Commons), it will be listed below.

Tracklist / Timecodes (Chapters)
"""

# =========================
# Utils
# =========================
def run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError("Command failed:\n" + " ".join(cmd) + "\n\n" + p.stdout)


def ffprobe_duration_sec(path: Path, ffprobe_bin: str = "ffprobe") -> float:
    cmd = [
        ffprobe_bin, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError("ffprobe failed for {}:\n{}".format(path, p.stderr))
    try:
        return float(p.stdout.strip())
    except Exception:
        return 0.0


def sec_to_hms(sec: float) -> str:
    s = int(max(0, sec))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return "{:02d}:{:02d}:{:02d}".format(h, m, ss)


def list_audio_files(in_dir: Path) -> List[Path]:
    return [p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]


def list_image_files(img_dir: Path) -> List[Path]:
    return [p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


# =========================
# Text candidates loader
# =========================
def load_text_candidates(path: Path) -> List[str]:
    """
    1行=1候補。空行と # コメント行は無視。
    行内の '\\n' は改行に変換して、2行テキストも作れるようにする。
    """
    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            f"サムネ文字候補ファイルが見つかりません: {path}\n"
            f"同じフォルダに {DEFAULT_THUMB_TEXT_FILE.name} を作成するか、"
            f"--thumb_text_candidates_file でパスを指定してください。"
        )

    lines = path.read_text(encoding="utf-8").splitlines()
    cands: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        cands.append(s.replace("\\n", "\n"))

    if not cands:
        raise RuntimeError(f"候補が0件です: {path}（空行/コメント以外を1行以上入れてください）")
    return cands


# =========================
# Output writer
# =========================
def write_title_and_description(
    out_dir: Path,
    chosen_text: Optional[str],
    ts_path: Path,
    title_map: Dict[str, str],
    header: str = DESCRIPTION_HEADER,
) -> Tuple[Path, Path]:
    """
    output/title.txt と output/description.txt を生成
    description = header + timestamps.txt の内容
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # title.txt
    if chosen_text and chosen_text.strip():
        key = chosen_text.strip()
        title = title_map.get(key)
        if not title:
            title = f"2 Hours {key} | Lyric-Free Background Music (No Lyrics)"
    else:
        title = "2 Hours Focus BGM | Lyric-Free Background Music (No Lyrics)"

    title_path = out_dir / "title.txt"
    title_path.write_text(title.strip() + "\n", encoding="utf-8")

    # description.txt
    if not ts_path.exists():
        raise RuntimeError(f"timestamps.txt が見つかりません: {ts_path}")
    ts_text = ts_path.read_text(encoding="utf-8").strip()
    if not ts_text:
        raise RuntimeError(f"timestamps.txt が空です: {ts_path}")

    # safety: first line should start with 00:00
    first_line = ts_text.splitlines()[0].strip()
    if not (first_line.startswith("00:00") or first_line.startswith("0:00")):
        ts_text = "00:00:00 Start\n" + ts_text

    theme_line = f"Theme: {chosen_text.strip()}\n\n" if (chosen_text and chosen_text.strip()) else ""
    description = (header.strip() + "\n\n" + theme_line + ts_text.strip() + "\n")

    desc_path = out_dir / "description.txt"
    desc_path.write_text(description, encoding="utf-8")

    return title_path, desc_path


# =========================
# Thumbnail: random pick + text overlay
# =========================
def pick_random_image(img_dir: Path, seed: Optional[int] = None) -> Path:
    files = list_image_files(img_dir)
    if not files:
        raise RuntimeError("背景画像が見つかりません: {}".format(img_dir))
    rng = random.Random(seed)
    return rng.choice(files)


def pick_random_images(img_dir: Path, k: int, seed: Optional[int] = None) -> List[Path]:
    files = list_image_files(img_dir)
    if not files:
        raise RuntimeError("背景画像が見つかりません: {}".format(img_dir))
    rng = random.Random(seed)
    if len(files) >= k:
        return rng.sample(files, k)
    return [rng.choice(files) for _ in range(k)]


def _load_font(fontfile: str, fontsize: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(fontfile, fontsize)
    except Exception:
        return ImageFont.load_default()


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int, int, int]:
    if hasattr(draw, "textbbox"):
        return draw.textbbox((0, 0), text, font=font, stroke_width=0)
    w, h = draw.textsize(text, font=font)
    return (0, 0, w, h)


def overlay_text_on_image(
    im_rgb: Image.Image,
    text: str,
    fontfile: str,
    fontsize: int,
    pos: str,
    margin: int,
    box_padding: int,
    box_alpha: int,
    stroke_width: int,
) -> Image.Image:
    im = im_rgb.convert("RGBA")
    draw = ImageDraw.Draw(im)

    font = _load_font(fontfile, fontsize)

    lines = text.split("\n")
    line_bboxes = [_text_bbox(draw, ln, font) for ln in lines]
    line_ws = [bb[2] - bb[0] for bb in line_bboxes]
    line_hs = [bb[3] - bb[1] for bb in line_bboxes]
    text_w = max(line_ws) if line_ws else 0
    text_h = sum(line_hs) + (len(lines) - 1) * int(fontsize * 0.20)

    W, H = im.size

    if pos == "top_left":
        x = margin
        y = margin
    elif pos == "top_right":
        x = W - margin - text_w
        y = margin
    elif pos == "bottom_left":
        x = margin
        y = H - margin - text_h
    elif pos == "bottom_right":
        x = W - margin - text_w
        y = H - margin - text_h
    elif pos == "center":
        x = (W - text_w) // 2
        y = (H - text_h) // 2
    else:
        x = margin
        y = H - margin - text_h

    # background box
    box = Image.new("RGBA", im.size, (0, 0, 0, 0))
    box_draw = ImageDraw.Draw(box)
    x0 = x - box_padding
    y0 = y - box_padding
    x1 = x + text_w + box_padding
    y1 = y + text_h + box_padding
    box_draw.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=int(fontsize * 0.35),
        fill=(0, 0, 0, box_alpha),
    )
    im = Image.alpha_composite(im, box)
    draw = ImageDraw.Draw(im)

    yy = y
    for i, ln in enumerate(lines):
        if not ln.strip():
            yy += int(fontsize * 0.9)
            continue
        draw.text(
            (x, yy),
            ln,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0, 255),
        )
        yy += line_hs[i] + int(fontsize * 0.20)

    return im.convert("RGB")


def pick_random_thumb_text(seed: int, candidates: List[str]) -> str:
    rng = random.Random(seed)
    return rng.choice(candidates)


def make_thumbnail(
    src_img: Path,
    out_path: Path,
    size: Tuple[int, int] = (1280, 720),
    add_text: bool = False,
    text_seed: int = 0,
    text_candidates: Optional[List[str]] = None,
    text_fixed: Optional[str] = None,
    text_fontfile: str = "/System/Library/Fonts/Supplemental/Arial.ttf",
    text_fontsize: int = 88,
    text_pos: str = "bottom_left",
    text_margin: int = 60,
    text_box_padding: int = 24,
    text_box_alpha: int = 110,
    text_stroke_width: int = 3,
) -> Tuple[Optional[str], Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chosen_text: Optional[str] = None
    if add_text:
        if text_fixed and text_fixed.strip():
            chosen_text = text_fixed.strip()
        else:
            if not text_candidates:
                raise RuntimeError("text_candidates が空です（候補ファイル読み込みに失敗している可能性）")
            chosen_text = pick_random_thumb_text(text_seed, text_candidates)

    with Image.open(src_img) as im:
        im = im.convert("RGB")
        thumb = ImageOps.fit(im, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

        if add_text and chosen_text:
            thumb = overlay_text_on_image(
                thumb,
                text=chosen_text,
                fontfile=text_fontfile,
                fontsize=text_fontsize,
                pos=text_pos,
                margin=text_margin,
                box_padding=text_box_padding,
                box_alpha=text_box_alpha,
                stroke_width=text_stroke_width,
            )

        thumb.save(out_path, format="JPEG", quality=92, optimize=True)

    return chosen_text, out_path


# =========================
# Audio mixing (NO hard trim)
# =========================
def make_mix_from_random_tracks(
    in_dir: Path,
    out_dir: Path,
    n: int = 50,
    xfade: float = 2.0,
    seed: Optional[int] = None,
    loudnorm: bool = True,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> Tuple[Path, Path, Path, float]:
    files = list_audio_files(in_dir)
    if not files:
        raise SystemExit("入力フォルダに音声ファイルが見つかりません")

    if seed is None:
        seed = int(datetime.now().strftime("%Y%m%d"))
    random.seed(seed)

    if len(files) >= n:
        picked = random.sample(files, n)
    else:
        picked = [random.choice(files) for _ in range(n)]
    random.shuffle(picked)

    work = out_dir / "_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    durations: List[float] = []
    for p in picked:
        try:
            d = ffprobe_duration_sec(p, ffprobe_bin=ffprobe_bin)
        except Exception:
            d = 0.0
        durations.append(d)

    starts: List[float] = [0.0]
    for i in range(1, len(picked)):
        prev = durations[i - 1]
        starts.append(starts[-1] + max(0.0, prev - xfade))

    current = work / "mix_000.wav"
    run([ffmpeg_bin, "-y", "-i", str(picked[0]), "-vn", "-ar", "48000", "-ac", "2", str(current)])

    for i in range(1, len(picked)):
        nxt = picked[i]
        out_tmp = work / "mix_{:03d}.wav".format(i)
        run([
            ffmpeg_bin, "-y",
            "-i", str(current),
            "-i", str(nxt),
            "-filter_complex",
            (
                "[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a0];"
                "[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a1];"
                "[a0][a1]acrossfade=d={}:c1=tri:c2=tri[a]".format(xfade)
            ),
            "-map", "[a]",
            str(out_tmp)
        ])
        current = out_tmp

    out_audio = out_dir / "mix.wav"
    if loudnorm:
        run([
            ffmpeg_bin, "-y", "-i", str(current),
            "-filter:a", "loudnorm=I=-14:TP=-1.5:LRA=11",
            "-ar", "48000", "-ac", "2",
            str(out_audio)
        ])
    else:
        run([ffmpeg_bin, "-y", "-i", str(current), "-ar", "48000", "-ac", "2", str(out_audio)])

    ts_path = out_dir / "timestamps.txt"
    with ts_path.open("w", encoding="utf-8") as f:
        for i, p in enumerate(picked):
            f.write("{} {}\n".format(sec_to_hms(starts[i]), p.stem))

    used_path = out_dir / "used_files.txt"
    with used_path.open("w", encoding="utf-8") as f:
        for p in picked:
            f.write(str(p) + "\n")

    mixed_dur = ffprobe_duration_sec(out_audio, ffprobe_bin=ffprobe_bin)
    return out_audio, ts_path, used_path, mixed_dur


# =========================
# Video (N images + crossfade + KenBurns + clock overlay)
# =========================
def build_filter_complex_for_slideshow(
    num_imgs: int,
    clip_dur: float,
    xfade: float,
    fps: int,
    width: int,
    height: int,
    kenburns_zoom_end: float,
    clock: bool,
    clock_fontfile: str,
    clock_fontsize: int,
    clock_margin: int,
    clock_scale: float,
    clock_left_shift: int,
) -> str:
    frames = int(round(clip_dur * fps))
    if frames < 1:
        frames = 1

    parts: List[str] = []

    for i in range(num_imgs):
        kb_end = kenburns_zoom_end if kenburns_zoom_end > 1.0 else 1.08
        z_inc = (kb_end - 1.0) / float(max(frames, 1))

        parts.append(
            "[{idx}:v]"
            "scale={w}:{h}:force_original_aspect_ratio=increase,"
            "crop={w}:{h},"
            "format=yuv420p,"
            "zoompan=z='min(zoom+{zinc:.8f},{zend:.3f})':"
            "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            "d={frames}:s={w}x{h}:fps={fps}"
            "[v{idx}]".format(
                idx=i, w=width, h=height,
                zinc=z_inc, zend=kb_end,
                frames=frames, fps=fps
            )
        )

    for i in range(num_imgs - 1):
        left = "[v0]" if i == 0 else "[x{}]".format(i - 1)
        right = "[v{}]".format(i + 1)
        out = "[x{}]".format(i)
        offset = (clip_dur - xfade) if i == 0 else (clip_dur - xfade) * float(i + 1)
        parts.append(
            "{l}{r}xfade=transition=fade:duration={d:.3f}:offset={o:.3f}{out}".format(
                l=left, r=right, d=xfade, o=offset, out=out
            )
        )

    final_label = "[x{}]".format(num_imgs - 2) if num_imgs >= 2 else "[v0]"

    if clock:
        # 経過時間を HH:MM:SS で表示（小数秒なし）
        clock_text = (
            "%{eif\\:trunc(t/3600)\\:d\\:2}\\:"
            "%{eif\\:trunc(mod(t\\,3600)/60)\\:d\\:2}\\:"
            "%{eif\\:trunc(mod(t\\,60))\\:d\\:2}"
        )

        # 位置：やや下へ（従来通り +40）
        y_pos = clock_margin + 40

        # ✅ 要望：時計を 2.5倍、少し左へ
        fs = max(1, int(clock_fontsize * float(clock_scale)))
        m = int(clock_margin + int(clock_left_shift))  # m を増やすほど左へ寄る（右端余白が増える）
        bw = max(1, int(12 * float(clock_scale)))      # boxborderw もフォントに合わせて拡大

        parts.append(
            "{inp}drawtext=fontfile='{font}':text='{text}':"
            "x=w-tw-{m}:y={y}:fontsize={fs}:fontcolor=white:"
            "box=1:boxcolor=black@0.35:boxborderw={bw}"
            "[vout]".format(
                inp=final_label, font=clock_fontfile, text=clock_text,
                m=m, y=y_pos, fs=fs, bw=bw
            )
        )
        return ";".join(parts)

    parts.append("{inp}copy[vout]".format(inp=final_label))
    return ";".join(parts)


def make_video_from_images_and_audio(
    images: List[Path],
    audio_path: Path,
    out_mp4: Path,
    xfade: float,
    fps: int,
    width: int,
    height: int,
    kenburns_zoom_end: float,
    clock: bool,
    clock_fontfile: str,
    clock_fontsize: int,
    clock_margin: int,
    clock_scale: float,
    clock_left_shift: int,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> None:
    if len(images) == 0:
        raise RuntimeError("画像が0枚です")

    audio_dur = ffprobe_duration_sec(audio_path, ffprobe_bin=ffprobe_bin)

    n = len(images)
    clip_dur = (audio_dur + (n - 1) * xfade) / float(n)
    if clip_dur <= xfade:
        clip_dur = xfade + 1.0

    filter_complex = build_filter_complex_for_slideshow(
        num_imgs=n,
        clip_dur=clip_dur,
        xfade=xfade,
        fps=fps,
        width=width,
        height=height,
        kenburns_zoom_end=kenburns_zoom_end,
        clock=clock,
        clock_fontfile=clock_fontfile,
        clock_fontsize=clock_fontsize,
        clock_margin=clock_margin,
        clock_scale=clock_scale,
        clock_left_shift=clock_left_shift,
    )

    cmd: List[str] = [ffmpeg_bin, "-y"]

    for img in images:
        cmd += ["-loop", "1", "-t", "{:.3f}".format(clip_dur), "-i", str(img)]

    cmd += ["-i", str(audio_path)]

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "{}:a".format(len(images)),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(out_mp4)
    ]

    run(cmd)


# =========================
# Main
# =========================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create mix.wav + timestamps.txt + thumbnail.jpg + title.txt + description.txt (+ optional video.mp4)"
    )

    ap.add_argument("--bgm_dir", default=DEFAULT_BGM_DIR)
    ap.add_argument("--img_dir", default=DEFAULT_IMG_DIR)
    ap.add_argument("--out_dir", default="./output")
    ap.add_argument("--seed", type=int, default=None, help="例: 20260105（未指定なら日付ベース）")

    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--audio_xfade", type=float, default=2.0, help="BGM同士のクロスフェード秒")
    ap.add_argument("--no_loudnorm", action="store_true")

    ap.add_argument("--thumb_w", type=int, default=1280)
    ap.add_argument("--thumb_h", type=int, default=720)

    # Thumbnail text options
    ap.add_argument("--thumb_text", action="store_true", help="サムネに文字を焼き込む（候補からランダム）")
    ap.add_argument("--thumb_text_fixed", default=None, help="固定文字（指定するとランダムではなくこれを使用）")
    ap.add_argument("--thumb_text_pos", default="bottom_left",
                    choices=["top_left", "top_right", "bottom_left", "bottom_right", "center"])
    ap.add_argument("--thumb_text_fontfile", default="/System/Library/Fonts/Supplemental/Arial.ttf")
    ap.add_argument("--thumb_text_size", type=int, default=88)
    ap.add_argument("--thumb_text_margin", type=int, default=60)
    ap.add_argument("--thumb_text_box_alpha", type=int, default=110)
    ap.add_argument("--thumb_text_box_padding", type=int, default=24)
    ap.add_argument("--thumb_text_stroke", type=int, default=3)

    # candidates file
    ap.add_argument("--thumb_text_candidates_file", default=str(DEFAULT_THUMB_TEXT_FILE),
                    help="サムネ文字候補（1行=1候補）のテキストファイルパス")

    # description header override (optional)
    ap.add_argument("--description_header_file", default=None,
                    help="descriptionヘッダーを外部ファイルで上書きしたい場合（英語テキスト）")

    # Video options
    ap.add_argument("--make_video", action="store_true", help="MP4動画も生成する（video.mp4）")
    ap.add_argument("--video_images", type=int, default=12, help="スライドショーに使う画像枚数（デフォルト12）")
    ap.add_argument("--video_xfade", type=float, default=1.0, help="画像切替のクロスフェード秒（デフォルト1.0）")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--kenburns_zoom_end", type=float, default=1.08, help="ズーム終点倍率（例: 1.06〜1.12）")

    ap.add_argument("--clock", action="store_true", help="右上に時計（経過時間 HH:MM:SS）を表示")
    ap.add_argument("--clock_fontfile", default="/System/Library/Fonts/Supplemental/Arial.ttf")
    ap.add_argument("--clock_fontsize", type=int, default=48)
    ap.add_argument("--clock_margin", type=int, default=40)

    # ✅ requested defaults: 2.5x bigger + slightly left
    ap.add_argument("--clock_scale", type=float, default=2.5, help="時計のフォント倍率（デフォルト2.5）")
    ap.add_argument("--clock_left_shift", type=int, default=40, help="時計を左へ寄せるpx（右端余白を増やす。デフォルト40）")

    ap.add_argument("--ffmpeg_bin", default="ffmpeg")
    ap.add_argument("--ffprobe_bin", default="ffprobe")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else int(datetime.now().strftime("%Y%m%d"))

    bgm_dir = Path(args.bgm_dir)
    img_dir = Path(args.img_dir)

    out_audio, ts_path, used_path, mixed_dur = make_mix_from_random_tracks(
        in_dir=bgm_dir,
        out_dir=out_dir,
        n=args.n,
        xfade=args.audio_xfade,
        seed=seed,
        loudnorm=(not args.no_loudnorm),
        ffmpeg_bin=args.ffmpeg_bin,
        ffprobe_bin=args.ffprobe_bin,
    )

    # ---- thumbnail ----
    picked_thumb = pick_random_image(img_dir, seed=seed)
    thumb_path = out_dir / "thumbnail.jpg"

    text_candidates: Optional[List[str]] = None
    if args.thumb_text and not (args.thumb_text_fixed and args.thumb_text_fixed.strip()):
        text_candidates = load_text_candidates(Path(args.thumb_text_candidates_file))

    text_seed = seed + 999
    chosen_text, _ = make_thumbnail(
        src_img=picked_thumb,
        out_path=thumb_path,
        size=(args.thumb_w, args.thumb_h),
        add_text=args.thumb_text,
        text_seed=text_seed,
        text_candidates=text_candidates,
        text_fixed=args.thumb_text_fixed,
        text_fontfile=args.thumb_text_fontfile,
        text_fontsize=args.thumb_text_size,
        text_pos=args.thumb_text_pos,
        text_margin=args.thumb_text_margin,
        text_box_padding=args.thumb_text_box_padding,
        text_box_alpha=max(0, min(255, args.thumb_text_box_alpha)),
        text_stroke_width=max(0, args.thumb_text_stroke),
    )

    # ---- title/description ----
    header = DESCRIPTION_HEADER
    if args.description_header_file:
        header_path = Path(args.description_header_file)
        if not header_path.exists():
            raise SystemExit(f"--description_header_file not found: {header_path}")
        header = header_path.read_text(encoding="utf-8").strip() + "\n"

    title_path, desc_path = write_title_and_description(
        out_dir=out_dir,
        chosen_text=chosen_text if args.thumb_text else None,
        ts_path=ts_path,
        title_map=TITLE_MAP,
        header=header,
    )

    # ---- video ----
    out_mp4 = out_dir / "video.mp4"
    picked_imgs: List[Path] = []
    if args.make_video:
        picked_imgs = pick_random_images(img_dir, k=args.video_images, seed=seed + 1)
        make_video_from_images_and_audio(
            images=picked_imgs,
            audio_path=out_audio,
            out_mp4=out_mp4,
            xfade=args.video_xfade,
            fps=args.fps,
            width=args.width,
            height=args.height,
            kenburns_zoom_end=args.kenburns_zoom_end,
            clock=args.clock,
            clock_fontfile=args.clock_fontfile,
            clock_fontsize=args.clock_fontsize,
            clock_margin=args.clock_margin,
            clock_scale=args.clock_scale,
            clock_left_shift=args.clock_left_shift,
            ffmpeg_bin=args.ffmpeg_bin,
            ffprobe_bin=args.ffprobe_bin,
        )

    print("DONE")
    print("Seed:", seed)
    print("Mix duration:", sec_to_hms(mixed_dur), "({:.1f}s)".format(mixed_dur))
    print("Audio:", out_audio)
    print("Timestamps:", ts_path)
    print("Used list:", used_path)
    print("Thumbnail src:", picked_thumb)
    print("Thumbnail:", thumb_path)
    if args.thumb_text:
        print("Thumbnail text:", chosen_text)
        if text_candidates is not None:
            print("Candidates file:", args.thumb_text_candidates_file)
    print("Title:", title_path)
    print("Description:", desc_path)

    if args.make_video:
        print("Video images ({}):".format(len(picked_imgs)))
        for p in picked_imgs:
            print(" -", p)
        print("Video:", out_mp4)
        if args.clock:
            print(f"Clock scale: {args.clock_scale}x")
            print(f"Clock left shift: {args.clock_left_shift}px")


if __name__ == "__main__":
    main()
