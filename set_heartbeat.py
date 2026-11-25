#!/usr/bin/env python3
"""Store the healthchecks.io ping URL in config.toml and send the first ping.

Usage (on the Pi):  cd ~/hamster && .venv/bin/python set_heartbeat.py
Leave it empty to turn the heartbeat off.
"""
import re
import subprocess
from pathlib import Path

CONFIG = Path(__file__).resolve().parent / "config.toml"


def main():
    url = input("貼上 healthchecks.io 的 Ping URL（直接按 Enter = 關閉斷線監控）: ").strip()
    if url and not re.fullmatch(r"https://[\w.-]+/[\w/-]+", url):
        print("看起來不是正確的網址（應該像 https://hc-ping.com/xxxx-xxxx），沒有變更。")
        return
    text = CONFIG.read_text(encoding="utf-8")
    line = f'url = "{url}"'
    if re.search(r"(?m)^\[heartbeat\]\s*$", text):
        text = re.sub(r"(?ms)^(\[heartbeat\]\s*\n(?:#.*\n)*)url\s*=.*$", lambda m: m.group(1) + line, text, count=1)
    else:
        text = text.rstrip("\n") + "\n\n[heartbeat]\n# 斷線監控 (healthchecks.io)；用 set_heartbeat.py 設定\n" + line + "\n"
    CONFIG.write_text(text, encoding="utf-8")
    if not url:
        print("已關閉斷線監控。")
        return
    r = subprocess.run(["sudo", "systemctl", "start", "hamster-heartbeat.service"])
    print("已儲存並送出第一次報平安，到 healthchecks.io 看看狀態有沒有變綠色。" if r.returncode == 0
          else "已儲存，但第一次報平安失敗，請確認網址和網路。")


if __name__ == "__main__":
    main()
