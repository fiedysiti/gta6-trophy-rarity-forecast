# GTA 6 Trophy Rarity Forecast

A program that forecasts the earn rate of all 55 Grand Theft Auto VI
trophies on Exophase at a future target date (26 November 2026, one week
after launch), as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

Grand Theft Auto VI launched on 19 November 2026. Its 55 trophies are
real: 1 Platinum, 2 Gold, 6 Silver, 46 Bronze, 28 of them hidden, all
single-player, no online trophy. The trophy list comes from Exophase's
public trophy page for the game.

## How to run

Python 3.13, standard library only, no network:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The program reads every `*.in` case in the input directory and writes
one `<id>.out` per case. Each output is a single JSON object:

```json
{"trophies": [{"id": 1, "q": [0.0153, 0.0589, 0.15, 0.3817, 1.4552]}, ...]}
```

55 rows, one per trophy, each `q` five numbers from 0 to 100 in
non-decreasing order, file under 65,536 bytes. Runs are deterministic:
two runs on the same case produce byte-identical output.

## Method in brief

A trophy's rate is a share of the players Exophase tracks, so it moves
under two opposing forces: players earn trophies (rate up) while new
players join tracking with few trophies (rate down — dilution). The
program:

1. Fits a least-squares trend of `logit(rate)` against `ln(1 + hours
   since launch)` on the reads from the 24 hours before the case
   checkpoint. A flat history (like launch night, when every read was
   identical) yields no fit, and the trend is not used.
2. Classifies each trophy from its description and tier (story chapter,
   final mission, grind, activity, money, ...) with keyword rules —
   never by trophy id — and looks up that class's prior median and
   ceiling for one week after launch.
3. Blends the trend projection with the class prior in logit space
   (trend weight 0.7 for a good fit, 0.45 otherwise, prior alone when
   there is no fit), clamps the result to the class ceiling, and
   interpolates toward the last observed rate for short horizons: the
   prior describes day 7, so a target hours after launch stays near
   the last read, tilted down by the dilution a refresh brings.
4. Widens the band honestly: spread = sqrt((0.5·√d)² + 0.8² + 0.4² +
   residual²) in logit units, where d is the horizon in days, 0.8 is
   the refresh shock, and 0.4 is class uncertainty. For the final case
   (d = 4.52) the spread is about 1.39 logit.

Class priors are calibrated from GTA 5 long-run tracker rates, scaled
down to a 7-day window. See `METHODOLOGY.md` for the full method, the
class table, the equations, and the limitations, and
`data/gta5_reference_rates.csv` for the GTA 5 analogues used.

## Example output (local practice run)

Running the program on the two practice cases from the competition kit
(`p00`, checkpoint 03:00 UTC 19 Nov, target 04:30 UTC) gives valid
output: 55 rows, quantiles non-decreasing, 2,787 bytes. Median vs
launch-night rate — every median sits below its launch-night rate,
because the forecast expects the dilution of the 04:30 refresh (the
first refresh grew tracked players from 638 to 3,868 and every rate
except the 100% one fell):

| Trophy | Launch rate | Median (q50) | Band (q05–q95) |
|---|---|---|---|
| Vice Got a Price! (Platinum) | 0.63% | 0.34% | 0.08–1.50% |
| Just Like Paradise (Gone Fishing) | 100% | 99.98% | 99.91–99.99% |
| Paradise Lost (Chapter 1) | 1.25% | 0.80% | 0.18–3.46% |
| You Gotta Know When to Get Off (final mission) | 1.10% | 0.67% | 0.15–2.91% |
| A Long Road (100% completion) | 0.63% | 0.36% | 0.08–1.58% |

For the actual target, 26 November, the same program forecasts the
Chapter 1 trophy at a 55% median, the final mission at 15%, 100%
completion at 1%, and the Platinum at 0.15% — the launch-night rates
(0.63–1.25%) are reviewer-inflated and will dilute down, not up.

## Honesty about the data

- Rates come from Exophase's tracked-player sample, which is
  self-selected (players who sync trophies). It is not the general
  player population; GTA 5 tracker rates run far above general-population
  rates, so the GTA 5 analogues are an upper bound, not a target.
- Exophase refreshes its counts on its own schedule (hours apart), not
  continuously; a refresh can move rates sharply, mostly down through
  dilution.
- The practice cases have no published answers, so no local loss can be
  computed; the practice run above only checks format, determinism,
  and sanity against the launch-night rates.
- The program never sees the future: it uses only each case's history
  before its checkpoint.

## Files

- `main.py` — the forecaster (the only entrypoint).
- `METHODOLOGY.md` — full method, calibration, assumptions, limits.
- `data/gta5_reference_rates.csv` — GTA 5 analogue rates used for
  calibration.
- `data/trophy_classes.csv` — the 55 GTA 6 trophies with their class,
  prior median, and ceiling.
