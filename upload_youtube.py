#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#python3 upload_youtube.py \
#  --client_secrets ./client_secret.json \
#  --video ./output/video.mp4 \
#  --thumb ./output/thumbnail.jpg \
#  --title_file ./output/title.txt \
#  --description_file ./output/description.txt \
#  --privacy public

import argparse
from pathlib import Path
from typing import Optional, List

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]

def get_youtube_service(client_secrets: Path, token_path: Path):
    creds = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    # トークンが古くてスコープ不足のときは再認証
    if creds and creds.scopes:
        if not set(SCOPES).issubset(set(creds.scopes)):
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES)
            creds = flow.run_local_server(
                port=0,
                prompt="consent",
                access_type="offline"
            )

        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    return build("youtube", "v3", credentials=creds)

def upload_video(
    youtube,
    video_file: Path,
    title: str,
    description: str,
    tags: Optional[List[str]],
    category_id: str,
    privacy_status: str,
    publish_at: Optional[str],
    made_for_kids: bool,
):
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": made_for_kids,
        }
    }

    if tags:
        body["snippet"]["tags"] = tags

    if publish_at:
        body["status"]["publishAt"] = publish_at

    media = MediaFileUpload(str(video_file), chunksize=1024 * 1024 * 8, resumable=True)

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"Uploading... {pct}%")

    video_id = response["id"]
    print("Uploaded video_id:", video_id)
    return video_id

def set_thumbnail(youtube, video_id: str, thumb_file: Path) -> bool:
    try:
        media = MediaFileUpload(str(thumb_file))
        youtube.thumbnails().set(videoId=video_id, media_body=media).execute()
        print("Thumbnail set.")
        return True

    except HttpError as e:
        print("\n[WARN] Failed to set thumbnail.")
        print("Reason:", getattr(e, "error_details", None) or str(e))

        if e.resp is not None and e.resp.status == 403:
            print(
                "\nYouTube側で『カスタムサムネイル』が有効化されていない可能性があります。\n"
                "対処: YouTube Studio → 設定 → チャンネル → 機能の利用要件 で電話番号認証を行い、\n"
                "カスタムサムネイルを有効化してから、再度 --video_id でサムネだけ設定してください。"
            )
        return False

def read_text_if_exists(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8").strip()

DEFAULT_TAGS = ",".join([
    # JP
    "作業用bgm", "勉強", "集中","bgm",
    # EN (translations)
    "work bgm", "study",
    # Recommended
    "focus", "deep work", "background music",
    "work music", "study music",
])

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--client_secrets", required=True, help="client_secret.json のパス")
    ap.add_argument("--token", default="./.yt_token/token.json", help="token保存先（初回認証後に自動保存）")

    # 動画ファイル or 既存 video_id
    ap.add_argument("--video", default=None, help="video.mp4 のパス（アップロードする場合）")
    ap.add_argument("--video_id", default=None, help="既存動画の video_id（サムネだけ設定する場合）")

    ap.add_argument("--thumb", default=None, help="thumbnail.jpg のパス（指定時のみ設定）")

    # ✅ NEW: title/description file (default: output/)
    ap.add_argument("--title_file", default=None, help="title.txt のパス（未指定なら --title を使用）")
    ap.add_argument("--description_file", default=None, help="description.txt のパス（未指定なら --description を使用）")

    # 手動指定も残す（互換維持）
    ap.add_argument("--title", default="", help="アップロード時のタイトル（--title_file が優先）")
    ap.add_argument("--description", default="", help="概要欄（--description_file が優先）")

    ap.add_argument("--tags", default=DEFAULT_TAGS, help="comma separated")
    ap.add_argument("--category_id", default="10", help="例: 10=Music, 27=Education など")
    ap.add_argument("--privacy", default="private", choices=["private", "unlisted", "public"])
    ap.add_argument("--publish_at", default=None, help="予約公開(RFC3339) 例: 2026-01-08T09:00:00+09:00")
    ap.add_argument("--made_for_kids", action="store_true", help="子ども向けにする場合のみ指定")

    args = ap.parse_args()

    client_secrets = Path(args.client_secrets)
    token_path = Path(args.token)

    thumb_file = Path(args.thumb) if args.thumb else None

    youtube = get_youtube_service(client_secrets, token_path)

    # ✅ NEW: file優先で title/description を確定
    title_from_file = read_text_if_exists(args.title_file)
    desc_from_file = read_text_if_exists(args.description_file)

    title = title_from_file if title_from_file is not None else (args.title.strip() if args.title else "")
    description = desc_from_file if desc_from_file is not None else (args.description if args.description else "")

    # 1) アップロードする
    video_id = args.video_id
    if args.video:
        video_file = Path(args.video)
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None

        if not title:
            raise SystemExit("Error: --video を使う場合は --title か --title_file を指定してください")

        video_id = upload_video(
            youtube=youtube,
            video_file=video_file,
            title=title,
            description=description,
            tags=tags,
            category_id=args.category_id,
            privacy_status=args.privacy,
            publish_at=args.publish_at,
            made_for_kids=args.made_for_kids,
        )

    if not video_id:
        raise SystemExit("Error: --video か --video_id のどちらかを指定してください")

    # 2) サムネ設定（失敗しても落とさない）
    if thumb_file:
        set_thumbnail(youtube, video_id, thumb_file)

if __name__ == "__main__":
    main()
