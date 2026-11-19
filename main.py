"""Forecast GTA 6 trophy earn rates on Exophase for a future target date.

Method in one paragraph: every trophy's rate is a share of the players
Exophase tracks, so it moves under two opposing forces -- players earn
trophies (rate up) while new players join tracking with few trophies
(rate down, dilution). This program projects each trophy's logit-rate
trend from its recent history, then blends that projection with a
class prior (calibrated from GTA 5 long-run analogues, scaled to one
week after launch) and clamps it to a class ceiling. The blend is
horizon-aware: the class prior describes day 7 (26 Nov), so a target
only hours after launch stays near the last observed rate, tilted down
by the dilution a refresh brings -- launch night proved that, when
tracked players jumped 638 -> 3868 at the first refresh, every rate
except the 100% one fell. The forecast band is deliberately wide: a
refresh of Exophase's counts can shift rates by roughly 0.8 logit, and
the pinball loss rewards honest bands.

Contract: read every *.in in MAKEFUN_INPUT_DIR (default /input), write
one <id>.out per case into MAKEFUN_OUTPUT_DIR (default /output) as
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.
Python 3.13 standard library only; no network, no randomness, fully
deterministic (output rows sorted by trophy id).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Launch moment; all time axes are hours since this instant.
RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)

# Class priors: (median percent at one week after launch, ceiling percent).
# Calibrated from GTA 5 long-run tracker rates, scaled down to a 7-day
# window (see METHODOLOGY.md). Medians are what a typical tracked
# player base reaches in the first week; ceilings are hard upper bounds
# for the trend projection.
CLASS_TABLE = {
    "intro_fishing": (90.0, 99.0),
    "story_ch1": (55.0, 70.0),
    "story_ch2": (45.0, 60.0),
    "story_ch3": (35.0, 50.0),
    "story_ch4": (25.0, 40.0),
    "story_final": (15.0, 25.0),
    "story_mission": (25.0, 40.0),
    "stranger_final": (10.0, 20.0),
    "completion_100": (1.0, 3.0),
    "platinum": (0.15, 0.5),
    "grind_silver": (2.5, 6.0),
    "activity_easy": (25.0, 45.0),
    "activity_mid": (12.0, 25.0),
    "activity_flavor": (15.0, 30.0),
    "money": (6.0, 15.0),
}

# Keyword rules, checked in order; the first match wins. Matching is on
# the trophy description (lowercased) plus tier, never on trophy id, so
# the classifier stays general if the list ever changes.
RULES = [
    ("complete gone fishing", "intro_fishing"),
    ("leonida legend", "platinum"),
    ("attain 100% completion", "completion_100"),
    ("complete chapter 1", "story_ch1"),
    ("complete chapter 2", "story_ch2"),
    ("complete chapter 3", "story_ch3"),
    ("complete chapter 4", "story_ch4"),
    ("complete the final mission for", "stranger_final"),
    ("complete the final mission", "story_final"),
    ("complete 250 stunts", "grind_silver"),
    ("earn a gold medal", "grind_silver"),
    ("complete 8 mission strands", "grind_silver"),
    ("complete 3 collectible card", "grind_silver"),
    ("waink it at 100", "grind_silver"),
    ("rob 5 bank vaults", "grind_silver"),
    ("try each type of activity", "activity_easy"),
    ("complete an arcade game", "activity_easy"),
    ("beat cal in a fishing challenge", "activity_easy"),
    ("earn $250,000 laundering", "money"),
    ("spend $100,000", "money"),
    ("purchase all garage", "money"),
    ("register 10 vehicles", "money"),
    ("complete breakfast run", "story_mission"),
    ("complete the cartel heist", "story_mission"),
    ("complete child's play", "story_mission"),
    ("complete headliners", "story_mission"),
    ("complete vcia takedown", "story_mission"),
    ("complete operation savage glory", "story_mission"),
    ("enjoy a cocktail", "activity_flavor"),
    ("smash your way out of a vehicle showroom", "activity_flavor"),
    ("follow-through at the kolt", "activity_flavor"),
    ("a partnership with mass appeal", "activity_flavor"),
    ("choices made", "activity_flavor"),
    ("second time's the charm", "activity_flavor"),
    ("what the hell just happened?", "activity_flavor"),
    ("bowl 'em over with a melee drive-by", "activity_flavor"),
]


def classify(trophy):
    """Return the class name for one trophy dict."""
    text = (trophy.get("description") or "").lower()
    if trophy.get("tier") == "platinum":
        return "platinum"
    for keyword, klass in RULES:
        if keyword in text:
            return klass
    return "activity_mid"


def clamp_pct(pct):
    return min(99.99, max(0.01, pct))


def to_logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def from_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round_half_up_4(value):
    """Round to 4 decimals, half up; whole numbers come back as int."""
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 10000.0
    return int(out) if float(out).is_integer() else out


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    return (moment - RELEASE).total_seconds() / 3600.0


def fit_logit_trend(case, index):
    """Least-squares fit of logit(rate) against ln(1 + hours since release).

    Only reads from the 24 hours before the case checkpoint are used.
    Returns (slope, intercept, residual_std, n_points) or None when the
    history cannot support a fit (too few distinct reads, or perfectly
    flat rates, as on launch night).
    """
    checkpoint = parse_time(case["at"])
    points = []
    seen_x = set()
    for read in case["history"]:
        moment = parse_time(read["at"])
        if (checkpoint - moment).total_seconds() > 24 * 3600:
            continue
        rate = read["rates"][index]
        if rate is None:
            continue
        x = math.log1p(max(0.0, hours_since_release(moment)))
        key = round(x, 6)
        if key in seen_x:
            continue
        seen_x.add(key)
        points.append((x, to_logit(rate)))
    if len(points) < 3:
        return None
    if len({round(y, 9) for _, y in points}) < 2:
        return None  # flat history: no slope can be estimated
    n = len(points)
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mean_x) ** 2 for p in points)
    if sxx <= 0:
        return None
    sxy = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    resid_sq = sum((p[1] - (intercept + slope * p[0])) ** 2 for p in points)
    resid = math.sqrt(resid_sq / max(1, n - 2))
    return slope, intercept, resid, n


def forecast_trophy(case, trophy, index):
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    klass = classify(trophy)
    prior_median, ceiling = CLASS_TABLE[klass]

    # How far into the seven-day launch window the target sits. The
    # class priors are calibrated for 26 Nov (day 7); a target only
    # hours after launch must stay near the last observed rate instead.
    week_fraction = min(1.0, max(0.0, hours_since_release(target) / (7 * 24)))

    # Long-horizon component: trend projection blended with the class
    # prior (the trend earns trust only with several reads and a real
    # slope; a flat history leaves the prior alone).
    fit = fit_logit_trend(case, index)
    if fit is None:
        long_logit = to_logit(prior_median)
        resid = 0.0
    else:
        slope, intercept, resid, n = fit
        x_target = math.log1p(max(0.0, hours_since_release(target)))
        projected = clamp_pct(from_logit(intercept + slope * x_target))
        projected = min(projected, ceiling)
        weight = 0.7 if (n >= 5 and abs(slope) > 0.05) else 0.45
        long_logit = weight * to_logit(projected) + (1.0 - weight) * to_logit(prior_median)

    # Short-horizon component: the last observed rate, tilted down by
    # the dilution a refresh brings. Launch night proved the point:
    # the first Exophase refresh grew tracked players 638 -> 3868 and
    # every rate except the 100% one fell, because newly tracked
    # players hold fewer trophies. The tilt fades as the target moves
    # deeper into the launch week and the prior takes over.
    last_rate = case["history"][-1]["rates"][index]
    if last_rate is None:
        last_rate = prior_median
    short_logit = to_logit(last_rate) - 0.6 * (1.0 - week_fraction)

    median_logit = (1.0 - week_fraction) * short_logit + week_fraction * long_logit

    # Band width in logit units: random-walk growth, a refresh shock
    # (Exophase recounts move rates ~0.8 logit), class uncertainty,
    # and whatever the trend fit left unexplained.
    spread = math.sqrt(
        (0.5 * math.sqrt(horizon_days)) ** 2 + 0.8 ** 2 + 0.4 ** 2 + resid ** 2
    )

    quantiles = []
    for z in Z_SCORES:
        q = round_half_up_4(clamp_pct(from_logit(median_logit + z * spread)))
        quantiles.append(q)
    # Rounding can tie adjacent quantiles; never let the row decrease.
    for i in range(1, len(quantiles)):
        if quantiles[i] < quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1]
    return {"id": trophy["id"], "q": quantiles}


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    rows = [
        forecast_trophy(case, trophy, i)
        for i, trophy in enumerate(case["trophies"])
    ]
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
