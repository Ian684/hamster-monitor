"""How dark the camera picture is, and where the AI's accuracy falls off (measured 2026-11-15).

Test: 40 photos the AI recognised clearly (>= 60 %) were made darker / grainier and run through the model:
  picture brightness (0-255)  66 -> 80 % recognised, 33 -> 57 %, 20 -> 42 %, 13 -> 18 %, 10 -> 8 %, <= 7 -> ~0 %
  extra grain (std, grey levels) 0 -> 100 %, 5 -> 75 %, 10 -> 25 %, 15 -> 2 %
Without night vision the camera first keeps the picture bright by turning up its gain (more grain), then the
picture gets darker. Either one alone is enough to lose the hamster.
"""
OK, DIM, DARK = "ok", "dim", "dark"
NAMES = {OK: "夠亮", DIM: "偏暗", DARK: "太暗"}
DIM_BRIGHT, DARK_BRIGHT = 35, 15     # mean grey level of the picture
DIM_GRAIN, DARK_GRAIN = 6, 10        # frame-to-frame noise (std, grey levels)


def level(bright, grain=None):
    """ok / dim (AI accuracy starts dropping) / dark (AI almost never finds the hamster)."""
    if bright is None:
        return None
    if bright < DARK_BRIGHT or (grain is not None and grain >= DARK_GRAIN):
        return DARK
    if bright < DIM_BRIGHT or (grain is not None and grain >= DIM_GRAIN):
        return DIM
    return OK
