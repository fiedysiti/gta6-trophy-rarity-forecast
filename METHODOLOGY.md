# Methodology — GTA 6 Trophy Rarity Forecast

## 1. What is being forecast

For each of Grand Theft Auto VI's 55 PlayStation 5 trophies, the earn
rate on Exophase — the percent of the players Exophase tracks for the
game who have earned that trophy — at the target `2026-11-26T00:00:00Z`,
one week after launch. The forecast is five quantiles per trophy
(5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale,
averaged over trophies and quantile levels. Lower loss is better.

The game launched 19 November 2026. The trophy list is real, recorded
from Exophase's public trophy page
(https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/):
1 Platinum, 2 Gold, 6 Silver, 46 Bronze; 28 hidden; all single-player;
no online trophy. The list is corroborated by press-start.com.au,
outputlag.com, leonidahub.org, and egamers.io.

## 2. Data sources

- **Competition kit cases.** Each case carries `history`: every read
  the house recorder made of the Exophase page before the case
  checkpoint, oldest first. Each read lists `players_tracked`,
  `trophies_earned`, and one rate per trophy. Launch night: 638
  players tracked, 943 trophies earned, rates unchanged from 23:50 UTC
  18 Nov to 04:00 UTC 19 Nov. First refresh 04:30 UTC: 3,868 players
  tracked, 4,276 trophies earned — every rate fell except the 100%
  one, because newly tracked players hold fewer trophies (dilution).
  Exophase refreshes on its own schedule, hours apart, not
  continuously.
- **GTA 5 analogue rates** (`data/gta5_reference_rates.csv`, from
  psn100.net long-run tracker rates and 6cheats). These are a
  **tracker-sample upper bound**, not a population target: GTA 5's
  Platinum sits at ~17–19% on trackers but ~5% in the general
  population. The calibration scales them down to a 7-day window.
- **Launch scale** (Newzoo, Sensor Tower, BBC, via press reports):
  roughly 37–51 million copies in week one, over 17 million digital
  pre-orders by 19 Nov. Tracked players will grow from 638 to the
  hundreds of thousands by 26 Nov — massive dilution for hard
  trophies, large gains for early-story trophies.

## 3. Model

Let `rate(t) = E(t) / N(t)`, earners over tracked players. Both grow;
`E` grows through player progression, `N` grows through new players
syncing (dilution). Work in logit space, `logit(p) = ln(p / (100-p))`
for a rate in percent, which keeps forecasts inside (0, 100) and
treats relative moves symmetrically.

### 3.1 Trend fit

For each trophy, least-squares fit of `logit(rate)` against
`x = ln(1 + hours since launch)` on the reads from the 24 hours before
the case checkpoint. The fit is used only when it exists: at least 3
distinct reads and non-flat rates. Launch-night history is perfectly
flat, so no fit is possible there — and none is claimed.

### 3.2 Trophy classification

Each trophy is classified at runtime from its description (lowercased
keyword matching, checked in order) plus tier — never by trophy id, so
the classifier stays general. Classes, prior medians, and ceilings for
one week after launch:

| Class | Prior median | Ceiling |
|---|---|---|
| intro_fishing (Complete Gone Fishing) | 90% | 99% |
| story_ch1 | 55% | 70% |
| story_ch2 | 45% | 60% |
| story_ch3 | 35% | 50% |
| story_ch4 | 25% | 40% |
| story_final (final mission) | 15% | 25% |
| story_mission (named mid-story missions) | 25% | 40% |
| stranger_final (final mission for Andy/Kenton/Billy/Randy/Tyrone) | 10% | 20% |
| completion_100 | 1% | 3% |
| platinum | 0.15% | 0.5% |
| grind_silver (250 stunts, gold shooting range, 8 stranger strands, 3 card sets, WAiNK 100, 5 bank vaults) | 2.5% | 6% |
| activity_easy (try each activity, arcade game, beat Cal fishing) | 25% | 45% |
| activity_mid (races, drone races, meditation, hunting, challenges, co-op, exploration) | 12% | 25% |
| activity_flavor (cocktail, showroom, bowling, and other light story beats) | 15% | 30% |
| money (launder $250k, spend $100k, garage properties, register vehicles) | 6% | 15% |

Unmatched descriptions fall back to `activity_mid`. The full
trophy-by-trophy mapping is in `data/trophy_classes.csv`.

Calibration logic (GTA 5 analogues, scaled to 7 days): GTA 5's first
story trophy sits near 55% long-run on trackers, its final mission
near 33–34%, 100% completion near 20–21% — but those are long-run
figures on a mature tracker sample after years, with a 150+ hour full
set. In one week, with a ~50-hour story and no mission replay, only a
small share of players finish: Chapter 1 ~55%, Chapter 4 ~25%, final
mission ~15%, 100% completion ~1%. The Platinum (~200 hours) sits at
~0.15%: the launch-night 0.63% is reviewer inflation (638 tracked
players, mostly reviewers and early players) and will dilute down, not
up. The easiest trophy, Gone Fishing, was already at 100% on launch
night and stays near 90%.

### 3.3 Blending

The long-horizon component blends the trend projection with the class
prior in logit space:

```
long = w * logit(trend projection at target) + (1 - w) * logit(prior median)
```

with `w = 0.7` for a good fit (at least 5 reads and |slope| > 0.05),
`w = 0.45` otherwise, and the prior alone when there is no fit. The
trend projection is clamped to the class ceiling — a logit-linear
trend without a clamp explodes for fast-rising trophies.

The prior describes day 7, so the final median interpolates by how
far into the launch week the target sits, `f = hours since launch at
target / 168`, clamped to [0, 1]:

```
short = logit(last observed rate) - 0.6 * (1 - f)
median = (1 - f) * short + f * long
```

The short-horizon component is persistence tilted down by the dilution
a refresh brings. Launch night proved the tilt: the first refresh grew
tracked players 638 → 3,868 and every rate except the 100% one fell.
The tilt fades as the target moves deeper into the week and the prior
takes over. For the graded target (26 Nov, f = 1) the median is the
long-horizon blend alone.

### 3.4 Spread

```
s = sqrt((0.5 * sqrt(d))^2 + 0.8^2 + 0.4^2 + resid^2)
```

in logit units, with `d` the horizon in days, 0.8 the refresh shock
(a refresh moves rates roughly 0.8 logit), 0.4 class uncertainty, and
`resid` the trend fit's residual standard deviation (0 with no fit).
For the final case (d = 4.52) s ≈ 1.39 — wide but honest; pinball
loss rewards honest bands. The house persistence baseline uses only
0.5·√d ≈ 1.06 and its 90% coverage on the practice cases was 0.

Quantiles: `q = inv_logit(logit(median) + z * s)` with
z = (−1.645, −0.674, 0, 0.674, 1.645), rounded to 4 decimals half-up,
clamped to [0.01, 99.99], forced non-decreasing after rounding.

## 4. Edge cases

- A trophy at 100% in the last read (Gone Fishing): its logit is
  clamped to 99.99 first; the median stays high (~90% at day 7) with
  a wide lower band, since new players do not all fish immediately.
- Platinum: median ~0.15% at day 7 — the launch-night 0.63% dilutes
  down.
- 100% completion: median ~1%; it needs 100–200 hours, almost nobody
  finishes in 7 days.
- Final mission: median ~15%; the story takes ~50 hours and only a
  small share of players finish within a week.

## 5. Validation

- Practice cases `p00`/`p01` (kit input): valid output — 55 rows,
  quantiles non-decreasing, under 65,536 bytes; two runs produce
  byte-identical output.
- Sanity: on the practice cases every median sits below its
  launch-night rate, as the dilution of the 04:30 refresh requires;
  bands are wide (e.g. Chapter 1: q05 0.18% – q95 3.46% around a 0.80%
  median).
- A synthetic case with the final horizon (checkpoint 21 Nov 11:30
  UTC, target 26 Nov) returns the class priors as medians (Platinum
  0.15%, Chapter 1 55%, final mission 15%, 100% completion 1%,
  Gone Fishing 90%), confirming the day-7 path.
- The practice cases have no published answers, so no local loss can
  be computed. No loss figure is claimed anywhere in this repo.

## 6. Assumptions and limitations

- Exophase's tracked sample is self-selected; rates are not
  population rates, and the GTA 5 analogues are an upper bound.
- Exophase's refresh schedule is not public; the model assumes a
  refresh before the target and that refreshes dilute rates, which
  launch night supports but later refreshes may not repeat exactly
  (as the tracked base grows, each refresh dilutes less).
- Class priors are judgment calls calibrated from one analogue game;
  if players progress much faster than GTA 5's pace (shorter story,
  more guides), story trophies could land higher and grind trophies
  lower, or vice versa.
- The trend fit uses only the last 24 hours of reads; longer history
  would stabilise the slope but risks stale regimes from launch night.
- One trophy is possibly missable (Positive Mental Attitude — the
  Chapter 1 balcony meditation); the model does not special-case it.
