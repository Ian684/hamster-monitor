#!/usr/bin/env python3
"""Hamster cage climate loop: DHT11 -> MAX7219 7-segment display + relay fan control.

Reads the DHT11 through the kernel IIO driver (dtoverlay=dht11,gpiopin=4),
shows "28.3C 63H" on the display, switches the fan with hysteresis, and logs
to SQLite plus a status.json snapshot for a future web dashboard.
"""
import glob
import json
import logging
import os
import signal
import sqlite3
import statistics
import sys
import time
import tomllib
from collections import deque
from datetime import datetime
from pathlib import Path

from gpiozero import OutputDevice
from luma.core.interface.serial import noop, spi
from luma.core.virtual import sevensegment
from luma.led_matrix.device import max7219

BASE = Path(__file__).resolve().parent
OVERRIDE_FILE = BASE / "data" / "fan_override.json"  # written by web.py
log = logging.getLogger("climate")


def read_override():
    """Manual fan command from the dashboard: {"mode": "on"|"off", "until": epoch}."""
    try:
        o = json.loads(OVERRIDE_FILE.read_text())
    except (OSError, ValueError):
        return None
    if o.get("mode") in ("on", "off") and float(o.get("until", 0)) > time.time():
        return o
    return None


def override_mtime():
    try:
        return OVERRIDE_FILE.stat().st_mtime_ns
    except OSError:
        return 0


def load_config():
    with open(BASE / "config.toml", "rb") as f:
        return tomllib.load(f)


def find_dht11():
    for name_file in glob.glob("/sys/bus/iio/devices/iio:device*/name"):
        if "dht11" in Path(name_file).read_text():
            return Path(name_file).parent
    raise RuntimeError("DHT11 IIO device not found (is dtoverlay=dht11 enabled?)")


def read_dht11(dev):
    """One reading, or None if the sensor glitched / returned nonsense."""
    try:
        t = int((dev / "in_temp_input").read_text()) / 1000
        h = int((dev / "in_humidityrelative_input").read_text()) / 1000
    except (OSError, ValueError):
        return None
    # DHT11 spec: 0-50 °C, 20-90 %RH; outside that is a bad read
    if not (0 < t <= 50 and 15 <= h <= 100):
        return None
    return t, h


class Fan:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dev = OutputDevice(cfg["gpio"], active_high=cfg["active_high"], initial_value=False)
        self.last_switch = 0.0
        self.had_override = False
        self.auto_on = False   # what automatic control wants (its own hysteresis memory)

    @property
    def on(self):
        return bool(self.dev.value)

    def update(self, t=None, h=None):
        """Apply a manual override from the dashboard if one is active, else the thresholds.

        t/h may be None when the sensor has no fresh reading; then only an override can act.
        Returns the active override dict (or None) for the status file.
        """
        c = self.cfg
        # The automatic decision is kept up to date all the time, also while a manual command is active, and
        # with its OWN memory for the in-between band (temp_off..temp_on). It used to start from the relay's
        # current state, so after "手動開" -> "自動" in that band the fan simply stayed on.
        if t is not None:
            if t >= c["temp_on"] or h >= c["hum_on"]:
                self.auto_on = True
            elif t <= c["temp_off"] and h <= c["hum_off"]:
                self.auto_on = False
        override = read_override()
        if override:
            want, manual = override["mode"] == "on", True
        elif t is not None:
            want, manual = self.auto_on, False
        else:
            return None
        # Leaving manual mode ("自動" pressed, or the 30 min ran out) is a user action too: apply the
        # automatic decision at once instead of waiting out the relay-protection delay.
        back_to_auto = self.had_override and not manual
        self.had_override = manual
        now = time.monotonic()
        # manual commands switch immediately; automatic ones respect min_switch_s
        if want != self.on and (manual or back_to_auto or now - self.last_switch >= c["min_switch_s"]):
            self.dev.value = want
            self.last_switch = now
            if manual:
                log.info("fan %s (manual override)", "ON" if want else "OFF")
            else:
                log.info("fan %s (t=%.1f h=%.0f)", "ON" if want else "OFF", t, h)
        return override

    def close(self):
        self.dev.close()  # pin released -> relay module pull-up keeps fan off


class Display:
    def __init__(self, cfg):
        self.dev = max7219(spi(port=0, device=0, gpio=noop()), cascaded=1)
        self.dev.contrast(cfg["brightness"])
        self.seg = sevensegment(self.dev)

    def show(self, text):
        self.seg.text = text[:16]

    def close(self):
        self.dev.cleanup()


class Store:
    def __init__(self, cfg):
        self.db_path = BASE / cfg["db"]
        self.status_path = BASE / cfg["status_file"]
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.db_path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS climate ("
            " ts TEXT PRIMARY KEY, temp REAL, hum REAL, fan INTEGER)"
        )
        self.db.commit()

    def record(self, t, h, fan):
        ts = datetime.now().isoformat(timespec="seconds")
        self.db.execute("INSERT OR REPLACE INTO climate VALUES (?,?,?,?)", (ts, t, h, int(fan)))
        self.db.commit()

    def status(self, **fields):
        fields["updated"] = datetime.now().isoformat(timespec="seconds")
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(fields, ensure_ascii=False))
        os.replace(tmp, self.status_path)

    def close(self):
        self.db.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    s_cfg, l_cfg = cfg["sensor"], cfg["log"]

    sensor = find_dht11()
    fan = Fan(cfg["fan"])
    display = Display(cfg["display"])
    store = Store(l_cfg)

    stop = False

    def handle_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    temps = deque(maxlen=s_cfg["median_window"])
    hums = deque(maxlen=s_cfg["median_window"])
    last_good = 0.0
    last_log = 0.0
    display.show("--.-C --H")
    log.info("started; fan thresholds %s", {k: v for k, v in cfg["fan"].items() if "_o" in k})

    try:
        while not stop:
            r = read_dht11(sensor)
            now = time.monotonic()
            if r:
                temps.append(r[0])
                hums.append(r[1])
                last_good = now

            if temps and now - last_good < s_cfg["stale_after_s"]:
                # calibration: [sensor] temp_offset / hum_offset = (reference thermometer) - (DHT11)
                t = round(statistics.median(temps) + s_cfg.get("temp_offset", 0.0), 1)
                h = min(100.0, max(0.0, statistics.median(hums) + s_cfg.get("hum_offset", 0.0)))
                override = fan.update(t, h)
                display.show(f"{t:4.1f}C {h:2.0f}H")
                store.status(temp=t, hum=h, fan=fan.on, ok=True, override=override)
                if now - last_log >= l_cfg["interval_s"]:
                    store.record(t, h, fan.on)
                    last_log = now
            else:
                # no valid data for a while: show error; fan keeps its last state unless overridden
                override = fan.update()
                display.show("SEnS Err")
                store.status(temp=None, hum=None, fan=fan.on, ok=False, override=override)
                if now - last_good >= s_cfg["stale_after_s"] and last_good:
                    log.warning("no valid DHT11 reading for %ds", int(now - last_good))

            # sleep in small steps so SIGTERM and dashboard fan commands are handled promptly
            end = time.monotonic() + s_cfg["interval_s"]
            ov_mtime = override_mtime()
            while not stop and time.monotonic() < end and override_mtime() == ov_mtime:
                time.sleep(0.2)
    finally:
        log.info("stopping")
        fan.close()
        display.close()
        store.close()


if __name__ == "__main__":
    sys.exit(main())
