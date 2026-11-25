#!/usr/bin/env python3
"""Set or clear the dashboard login password. Only a salted hash is stored in config.toml.

Usage (on the Pi):  cd ~/hamster && .venv/bin/python set_password.py           login password (when no allow_ips)
                    cd ~/hamster && .venv/bin/python set_password.py --edit    public link (ngrok): can edit
                    cd ~/hamster && .venv/bin/python set_password.py --judge   public link (ngrok): read-only, for judges
Leave the password empty to remove it (removing both tunnel passwords closes the public link).
"""
import getpass
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

CONFIG = Path(__file__).resolve().parent / "config.toml"


KEYS = {"--edit": ("tunnel_edit_password_hash", "外部連線（可編輯）"), "--judge": ("judge_password_hash", "外部連線（唯讀，給評審）")}


def main():
    key, what = KEYS.get(sys.argv[1] if len(sys.argv) > 1 else "", ("password_hash", "登入"))
    pw = getpass.getpass(f"{what}的新密碼 (直接按 Enter = 取消這組密碼): ")
    if pw:
        if getpass.getpass("再輸入一次: ") != pw:
            print("兩次輸入不一樣，沒有變更。")
            return
        if len(pw) < 6:
            print("密碼至少 6 個字元，沒有變更。")
            return
        web = tomllib.loads(CONFIG.read_text(encoding="utf-8")).get("web", {})
        other = {"tunnel_edit_password_hash": "judge_password_hash", "judge_password_hash": "tunnel_edit_password_hash"}.get(key)
        if other and web.get(other) and check_password_hash(web[other], pw):
            print("可編輯和唯讀不能用同一組密碼，沒有變更。")
            return
        value = generate_password_hash(pw)
    else:
        value = ""

    text = CONFIG.read_text(encoding="utf-8")
    line = f'{key} = "{value}"'
    pat = rf"(?m)^{key}\s*=.*$"
    if re.search(pat, text):
        text = re.sub(pat, lambda _: line, text)
    else:
        text = re.sub(r"(?m)^\[web\]\s*$", lambda m: m.group(0) + "\n" + line, text, count=1)
    CONFIG.write_text(text, encoding="utf-8")

    r = subprocess.run(["sudo", "systemctl", "restart", "hamster-web"])
    if r.returncode == 0:
        print(f"已設定{what}密碼，網頁已重新啟動。" if value else f"已取消{what}密碼，網頁已重新啟動。")
    else:
        print("設定已存檔，但重啟網頁失敗，請執行: sudo systemctl restart hamster-web")


if __name__ == "__main__":
    main()
