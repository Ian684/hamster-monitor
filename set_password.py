#!/usr/bin/env python3
"""Set or clear the dashboard login password. Only a salted hash is stored in config.toml.

Usage (on the Pi):  cd ~/hamster && .venv/bin/python set_password.py
Leave the password empty to remove the login requirement.
"""
import getpass
import re
import subprocess
from pathlib import Path

from werkzeug.security import generate_password_hash

CONFIG = Path(__file__).resolve().parent / "config.toml"


def main():
    pw = getpass.getpass("新密碼 (直接按 Enter = 取消登入密碼): ")
    if pw:
        if getpass.getpass("再輸入一次: ") != pw:
            print("兩次輸入不一樣，沒有變更。")
            return
        if len(pw) < 6:
            print("密碼至少 6 個字元，沒有變更。")
            return
        value = generate_password_hash(pw)
    else:
        value = ""

    text = CONFIG.read_text(encoding="utf-8")
    line = f'password_hash = "{value}"'
    if re.search(r"(?m)^password_hash\s*=.*$", text):
        text = re.sub(r"(?m)^password_hash\s*=.*$", lambda _: line, text)
    else:
        text = re.sub(r"(?m)^\[web\]\s*$", lambda m: m.group(0) + "\n" + line, text, count=1)
    CONFIG.write_text(text, encoding="utf-8")

    r = subprocess.run(["sudo", "systemctl", "restart", "hamster-web"])
    if r.returncode == 0:
        print("已設定登入密碼，網頁已重新啟動。" if value else "已取消登入密碼，網頁已重新啟動。")
    else:
        print("設定已存檔，但重啟網頁失敗，請執行: sudo systemctl restart hamster-web")


if __name__ == "__main__":
    main()
