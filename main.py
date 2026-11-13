#!/usr/bin/env python3
"""make.fun entry for polymarket-nvidia-ceo-forecast-01: forecast, minute by
minute and 4 hours ahead, the Yes price of every market of Polymarket's
"Next CEO of NVIDIA?" event, as five quantiles a minute.

Method: empirical calibration instead of a fixed normal spread. Prices in
this event move in steps -- long flat stretches, small ticks, occasional
jumps -- so for each case this program measures the distribution of h-minute
price changes in the case's own 72 hours of history and reads the forecast
quantiles off that distribution:

  * changes are normalised by a per-market scale (the root-mean-square of
    its recent one-minute moves), so quiet and jumpy markets each get the
    spread they actually showed;
  * normalised changes are pooled across the case's markets, and each
    market's own empirical quantiles are shrunk toward the pooled ones in
    proportion to how much history the market has, so a market that started
    an hour ago still gets a sane, mostly pooled, forecast;
  * quantiles are computed exactly at a set of horizon knots and
    interpolated in sqrt(h) between them, the way price-move spreads grow;
  * every quantile is clamped to [0, 1]: a price can never leave it.

Everything is a deterministic function of the case file: no randomness, no
clock, no network, standard library only, so two runs of a case write the
same bytes.
"""

import json
import math
import os
import sys

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.644854, -0.67449, 0.0, 0.67449, 1.644854)

PARAMS = {
    # Minutes of history (counted back from the checkpoint) whose h-step
    # changes form the empirical sample. Tuned on the practice backtest:
    # a few hours, not the whole 72 -- the market's activity regime turns
    # over, and older samples describe a livelier market than the next
    # 4 hours turn out to be. The loss is flat between 400 and 500.
    "hist_minutes": 420,
    # Minutes of history over which a market's scale (RMS one-minute move)
    # is measured; shorter = reacts faster to the current activity regime.
    "scale_minutes": 360,
    # Horizons at which change quantiles are measured exactly; between
    # knots they are interpolated linearly in sqrt(h).
    "knots": (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 240),
    # Shrinkage of a market's own quantiles toward the pooled ones:
    # weight on its own sample is n_windows / (n_windows + shrink_k).
    # Tuned value is large: with only a few hundred sample windows a
    # market's own quantiles are mostly noise, so the forecast is the
    # pooled shape, particularised by the market's scale alone.
    "shrink_k": 20000.0,
    # Shrinkage of a market's scale toward the pooled scale, by minutes of
    # scale history: weight minutes / (minutes + scale_shrink_k).
    "scale_shrink_k": 120.0,
    # A market's scale never falls below this fraction of the pooled scale:
    # a market that has not moved yet can still jump.
    "scale_floor_frac": 0.2,
    # Drift: the market's mean one-minute move over the last drift_minutes,
    # kept with this weight. The practice backtest rejects drift in both
    # directions (following the trend and fading it each cost far more
    # than they earn, because the term grows with h): the median stays at
    # the checkpoint price.
    "drift_weight": 0.0,
    "drift_minutes": 360,
    # Global width calibration: every quantile offset is multiplied by
    # this. The backtest's optimum is 1.0 -- at the tuned windows the
    # empirical distribution is already calibrated -- so it stays.
    "width": 1.0,
}


def _quantile(sorted_vals, tau):
    """The tau quantile of an ascending list, linear interpolation."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 >= n:
        return sorted_vals[-1]
    a = sorted_vals[lo]
    return a + frac * (sorted_vals[lo + 1] - a)


def _price6(v):
    """Clamp to [0, 1] and round to 6 decimals (never -0.0)."""
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    return round(v * 1e6) / 1e6 + 0.0


def _market_scale(prices, scale_minutes):
    """RMS of the one-minute moves over the last scale_minutes minutes.

    Returns (scale, n_moves)."""
    n = len(prices)
    if n < 2:
        return 0.0, 0
    lo = max(0, n - 1 - scale_minutes)
    ss = 0.0
    cnt = 0
    prev = prices[lo]
    for i in range(lo + 1, n):
        d = prices[i] - prev
        prev = prices[i]
        ss += d * d
        cnt += 1
    if cnt == 0:
        return 0.0, 0
    return math.sqrt(ss / cnt), cnt


def forecast_case(case, params):
    """The output object {"forecasts": [...]} for one parsed case."""
    horizon = int(case["horizon_minutes"])
    markets = case["markets"]
    knots = [h for h in params["knots"] if h <= horizon]
    if not knots or knots[-1] != horizon:
        knots.append(horizon)
    sqrt_knots = [math.sqrt(h) for h in knots]

    # --- per-market scales, shrunk toward the pooled scale -----------------
    raw_scales = []
    pooled_ss = 0.0
    pooled_cnt = 0
    for m in markets:
        s, cnt = _market_scale(m["prices"], params["scale_minutes"])
        raw_scales.append((s, cnt))
        pooled_ss += s * s * cnt
        pooled_cnt += cnt
    pooled_scale = math.sqrt(pooled_ss / pooled_cnt) if pooled_cnt else 0.0
    k_s = params["scale_shrink_k"]
    floor = params["scale_floor_frac"] * pooled_scale
    scales = []
    for s, cnt in raw_scales:
        if pooled_scale <= 0.0:
            scales.append(0.0)
        else:
            w = cnt / (cnt + k_s) if cnt else 0.0
            shrunk = w * s + (1.0 - w) * pooled_scale
            scales.append(shrunk if shrunk > floor else floor)

    # --- empirical change quantiles at the knots ---------------------------
    # Per knot: the pooled sorted normalised changes, and per market its own
    # sorted normalised changes with their count.
    hist = params["hist_minutes"]
    pooled_sorted = []
    per_market = []  # per market: list over knots of (sorted list, count)
    for ki, h in enumerate(knots):
        pooled = []
        for mi, m in enumerate(markets):
            prices = m["prices"]
            seg = prices[-hist:] if len(prices) > hist else prices
            inv = 1.0 / scales[mi] if scales[mi] > 0.0 else 0.0
            if inv > 0.0 and len(seg) > h:
                changes = [(b - a) * inv for a, b in zip(seg, seg[h:])]
            else:
                changes = []
            changes.sort()
            if ki == 0:
                per_market.append([])
            per_market[mi].append((changes, len(changes)))
            pooled.extend(changes)
        pooled.sort()
        pooled_sorted.append(pooled)

    # --- drift (kept only with drift_weight > 0) ---------------------------
    drifts = []
    dw = params["drift_weight"]
    dmin = params["drift_minutes"]
    for m in markets:
        prices = m["prices"]
        if dw != 0.0 and len(prices) >= 2:
            seg = prices[-(dmin + 1):]
            drifts.append(dw * (seg[-1] - seg[0]) / (len(seg) - 1))
        else:
            drifts.append(0.0)

    # --- assemble the rows --------------------------------------------------
    k_q = params["shrink_k"]
    forecasts = []
    for mi, m in enumerate(markets):
        x = m["prices"][-1] if m["prices"] else 0.5
        s = scales[mi]
        drift = drifts[mi]
        # Quantile offsets (price units) at each knot.
        knot_rows = []
        for ki in range(len(knots)):
            pooled = pooled_sorted[ki]
            own, cnt = per_market[mi][ki]
            w = cnt / (cnt + k_q) if cnt else 0.0
            row = []
            width = params["width"]
            for tau in QUANTILES:
                qp = _quantile(pooled, tau) if pooled else 0.0
                qo = _quantile(own, tau) if own else qp
                row.append(
                    width * s * (w * qo + (1.0 - w) * qp) + drift * knots[ki]
                )
            knot_rows.append(row)
        rows = []
        ki = 0
        for h in range(1, horizon + 1):
            while ki < len(knots) - 2 and knots[ki + 1] < h:
                ki += 1
            if h == knots[ki]:
                offs = knot_rows[ki]
            elif h < knots[0]:
                offs = knot_rows[0]
            elif h > knots[-1]:
                offs = knot_rows[-1]
            else:
                f = (math.sqrt(h) - sqrt_knots[ki]) / (
                    sqrt_knots[ki + 1] - sqrt_knots[ki]
                )
                a = knot_rows[ki]
                b = knot_rows[ki + 1]
                offs = [a[j] + f * (b[j] - a[j]) for j in range(len(QUANTILES))]
            row = []
            run = 0.0
            for j in range(len(QUANTILES)):
                v = _price6(x + offs[j])
                if v < run:
                    v = run
                else:
                    run = v
                row.append(v)
            rows.append(row)
        forecasts.append({"id": m["id"], "q": rows})
    return {"forecasts": forecasts}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(fn for fn in os.listdir(input_dir) if fn.endswith(".in"))
    for fn in names:
        with open(os.path.join(input_dir, fn), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        out = forecast_case(case, PARAMS)
        path = os.path.join(output_dir, fn[: -len(".in")] + ".out")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"forecasts": out["forecasts"]}))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
