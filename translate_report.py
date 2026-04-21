"""
translate_report.py
找到 reports/ 目錄下最新的子資料夾，將 complete_report.md 用 Gemini API
翻譯成繁體中文並輸出為 HTML，然後用 Chrome 開啟。
"""

import os
import sys
import subprocess
import pathlib
import textwrap
import json
import time
import random
import hashlib

# 強制 stdout/stderr 使用 UTF-8（Windows 終端機相容）
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv

# ── 設定 ──────────────────────────────────────────────────────────────────────
REPORTS_DIR = pathlib.Path(__file__).parent / "reports"
ENV_FILE    = pathlib.Path(__file__).parent / ".env"
MODEL_NAME  = "gemini-2.0-flash"   # 翻譯用模型（穩定版，速度快）
CHUNK_CHARS = 12000                 # 每次送翻譯的字元上限
CHECKPOINT_FILE_NAME = "complete_report_zh.progress.json"
MAX_RETRIES = 8
INITIAL_RETRY_DELAY = 5
MAX_RETRY_DELAY = 90

# ── 載入 API Key ──────────────────────────────────────────────────────────────
load_dotenv(ENV_FILE)
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("[ERROR] 找不到 GOOGLE_API_KEY，請確認 .env 設定。")
    sys.exit(1)
genai.configure(api_key=api_key)

# ── 找最新子資料夾 ─────────────────────────────────────────────────────────────
def find_latest_report_dir() -> pathlib.Path:
    subdirs = [d for d in REPORTS_DIR.iterdir() if d.is_dir()]
    if not subdirs:
        print("[ERROR] reports/ 目錄下找不到任何子資料夾。")
        sys.exit(1)
    return max(subdirs, key=lambda d: d.stat().st_mtime)

# ── 分段與 checkpoint ────────────────────────────────────────────────────────
def split_into_chunks(text: str) -> list[str]:
    """將長文本切成若干段落，避免單次請求過大。"""
    lines = text.split("\n")
    chunks, current = [], []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1
        if current_len + line_len > CHUNK_CHARS and current:
            chunks.append("\n".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += line_len
    if current:
        chunks.append("\n".join(current))
    return chunks


def build_source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_checkpoint(
    checkpoint_file: pathlib.Path,
    *,
    source_hash: str,
    total_chunks: int,
) -> list[str]:
    if not checkpoint_file.exists():
        return []

    try:
        payload = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print("  [WARN] checkpoint 讀取失敗，將從頭重新翻譯。")
        return []

    if payload.get("source_hash") != source_hash:
        print("  [INFO] 原始報告內容已變更，忽略舊 checkpoint。")
        return []
    if payload.get("model_name") != MODEL_NAME:
        print("  [INFO] 翻譯模型已變更，忽略舊 checkpoint。")
        return []
    if payload.get("chunk_chars") != CHUNK_CHARS:
        print("  [INFO] 分段設定已變更，忽略舊 checkpoint。")
        return []

    translated_parts = payload.get("translated_parts")
    if not isinstance(translated_parts, list):
        return []

    if len(translated_parts) > total_chunks:
        print("  [WARN] checkpoint 與目前分段數不一致，將從頭重新翻譯。")
        return []

    return translated_parts


def save_checkpoint(
    checkpoint_file: pathlib.Path,
    *,
    source_hash: str,
    total_chunks: int,
    translated_parts: list[str],
) -> None:
    payload = {
        "source_hash": source_hash,
        "model_name": MODEL_NAME,
        "chunk_chars": CHUNK_CHARS,
        "total_chunks": total_chunks,
        "completed_chunks": len(translated_parts),
        "translated_parts": translated_parts,
    }
    checkpoint_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def is_retryable_error(exc: Exception) -> bool:
    retryable_types = (
        google_exceptions.ResourceExhausted,
        google_exceptions.TooManyRequests,
        google_exceptions.ServiceUnavailable,
        google_exceptions.DeadlineExceeded,
        google_exceptions.InternalServerError,
    )
    if isinstance(exc, retryable_types):
        return True

    code = getattr(exc, "code", None)
    if callable(code):
        try:
            code = code()
        except TypeError:
            code = None
    return code in {429, 500, 503, 504}


def generate_content_with_retry(prompt: str, model, *, chunk_index: int, total_chunks: int):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return model.generate_content(prompt)
        except Exception as exc:
            if not is_retryable_error(exc) or attempt == MAX_RETRIES:
                raise

            delay = min(MAX_RETRY_DELAY, INITIAL_RETRY_DELAY * (2 ** (attempt - 1)))
            delay += random.uniform(0, 1)
            print(
                f"  [WARN] 第 {chunk_index}/{total_chunks} 段遇到暫時性錯誤 "
                f"({exc.__class__.__name__})，{delay:.1f} 秒後重試 "
                f"({attempt}/{MAX_RETRIES})..."
            )
            time.sleep(delay)


# ── 分段翻譯 ──────────────────────────────────────────────────────────────────
def translate_chunks(text: str, model, report_dir: pathlib.Path) -> str:
    """將長文本切成若干段落各自翻譯後合併，並保存進度以便重跑續傳。"""
    chunks = split_into_chunks(text)
    total = len(chunks)
    source_hash = build_source_hash(text)
    checkpoint_file = report_dir / CHECKPOINT_FILE_NAME

    translated_parts = load_checkpoint(
        checkpoint_file,
        source_hash=source_hash,
        total_chunks=total,
    )

    if translated_parts:
        print(f"  [INFO] 從 checkpoint 接續，已完成 {len(translated_parts)}/{total} 段。")

    for i, chunk in enumerate(chunks[len(translated_parts):], len(translated_parts) + 1):
        print(f"  翻譯中... ({i}/{total})", end="\r", flush=True)
        prompt = textwrap.dedent(f"""
            請將以下 Markdown 格式的股票分析報告翻譯成繁體中文。
            規則：
            - 保留所有 Markdown 語法（標題 #、粗體 **、表格 |、清單 - 等）
            - 專業術語可保留英文原文，例如 RSI、MACD、EMA、SMA、ATR、VWMA
            - 公司名稱、股票代號保留英文，例如 SOFI、SoFi Technologies
            - 只輸出翻譯結果，不要加任何說明或前後綴

            ---
            {chunk}
        """).strip()
        response = generate_content_with_retry(
            prompt,
            model,
            chunk_index=i,
            total_chunks=total,
        )
        translated_parts.append(response.text)
        save_checkpoint(
            checkpoint_file,
            source_hash=source_hash,
            total_chunks=total,
            translated_parts=translated_parts,
        )

    print()  # 換行
    return "\n\n".join(translated_parts)

# ── Markdown → HTML ───────────────────────────────────────────────────────────
def md_to_html(md_text: str, title: str) -> str:
    import markdown as md_lib
    body = md_lib.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br"]
    )
    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    body {{
      font-family: "Microsoft JhengHei", "PingFang TC", "Noto Sans TC", sans-serif;
      max-width: 960px;
      margin: 40px auto;
      padding: 0 24px 60px;
      line-height: 1.8;
      color: #1a1a2e;
      background: #f8f9fa;
    }}
    h1 {{ font-size: 2em; border-bottom: 3px solid #2d6a4f; padding-bottom: 10px; color: #1b4332; }}
    h2 {{ font-size: 1.5em; border-bottom: 2px solid #74c69d; padding-bottom: 6px;
          color: #2d6a4f; margin-top: 40px; }}
    h3 {{ font-size: 1.2em; color: #40916c; margin-top: 28px; }}
    h4 {{ font-size: 1em; color: #52b788; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 20px 0;
      background: #fff;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 1px 4px rgba(0,0,0,.08);
    }}
    th {{
      background: #2d6a4f;
      color: #fff;
      padding: 10px 14px;
      text-align: left;
    }}
    td {{ padding: 9px 14px; border-bottom: 1px solid #e9ecef; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:nth-child(even) {{ background: #f1f8f4; }}
    strong {{ color: #1b4332; }}
    blockquote {{
      border-left: 4px solid #74c69d;
      margin: 16px 0;
      padding: 8px 16px;
      background: #e9f5ee;
      border-radius: 0 6px 6px 0;
    }}
    code {{
      background: #e9ecef;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: .9em;
    }}
    hr {{ border: none; border-top: 2px solid #dee2e6; margin: 32px 0; }}
    a {{ color: #2d6a4f; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""

# ── 開啟 Chrome ───────────────────────────────────────────────────────────────
def open_in_chrome(html_path: pathlib.Path):
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for chrome in candidates:
        if pathlib.Path(chrome).exists():
            subprocess.Popen([chrome, str(html_path)])
            return
    # Fallback: 用系統預設瀏覽器
    os.startfile(str(html_path))

# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    report_dir = find_latest_report_dir()
    md_file = report_dir / "complete_report.md"
    checkpoint_file = report_dir / CHECKPOINT_FILE_NAME

    if not md_file.exists():
        print(f"[ERROR] 找不到 {md_file}")
        sys.exit(1)

    print(f"\n[翻譯報告] {report_dir.name}")
    md_text = md_file.read_text(encoding="utf-8")

    model = genai.GenerativeModel(MODEL_NAME)

    print("[1/3] 翻譯中（繁體中文）...")
    translated_md = translate_chunks(md_text, model, report_dir)

    print("[2/3] 轉換為 HTML...")
    title = f"交易分析報告 — {report_dir.name}"
    html_content = md_to_html(translated_md, title)

    out_file = report_dir / "complete_report_zh.html"
    out_file.write_text(html_content, encoding="utf-8")
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    print(f"[3/3] 已儲存：{out_file}")

    print("      用 Chrome 開啟中...")
    open_in_chrome(out_file)
    print("完成。\n")


if __name__ == "__main__":
    main()
