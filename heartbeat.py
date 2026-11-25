#!/usr/bin/env python3
"""Heartbeat ping for healthchecks.io, run every 5 minutes by hamster-heartbeat.timer.

If the Pi loses power or network, the pings stop and healthchecks.io notifies you.
If the Pi is up but a key service is down, it pings the /fail endpoint with the reason.
"""
import subprocess
import sys
import tomllib
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
SERVICES = {"hamster-climate": "溫濕度與風扇", "hamster-web": "網頁", "hamster-alerts": "通知"}


def main():
    with open(BASE / "config.toml", "rb") as f:
        url = tomllib.load(f).get("heartbeat", {}).get("url", "").strip()
    if not url:
        return 0  # not configured yet
    down = [name for unit, name in SERVICES.items()
            if subprocess.run(["systemctl", "is-active", unit], capture_output=True, text=True).stdout.strip() != "active"]
    target, body = (url + "/fail", ("停止的服務：" + "、".join(down)).encode()) if down else (url, b"ok")
    for attempt in range(3):
        try:
            urllib.request.urlopen(urllib.request.Request(target, data=body, method="POST"), timeout=10).read()
            return 0
        except OSError as e:
            err = e
    print(f"ping failed: {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
