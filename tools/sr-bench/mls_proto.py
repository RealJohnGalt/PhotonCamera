#!/usr/bin/env python3
"""Moving-least-squares prototype for the SR drizzle inverse.

The shipped drizzle is a normalized local average over the deposited samples
(MLS degree-0): it cannot invert the sampling above ~0.32 cyc/out, where the
band is moire-dominated. This fits a per-output-pixel regularized
least-squares local model (degree-0/1/2 polynomial) over the SAME irregular
samples and scores it against the aperture-filtered truth with the same
band metrics burst_bench uses - an apples-to-apples test of whether a true
irregular-sampling inverse recovers real detail the local average cannot.

Usage:
    python3 mls_proto.py                      # tones 9f/40f, degrees 0/1/2
    python3 mls_proto.py --frames 40 --lambdas 1e-4,1e-3,1e-2 --radius 2.5
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import sr_bench as SB  # noqa: E402
from burst_bench import (build_scene, sample_scene, scene_luma, fft_band,
                         struct_field, ramp_field)  # noqa: E402


class Args:
    pass


def collect(args, fx, fy, ph, amp, mtf, rng):
    """Irregular samples: per raw site per frame, (position, value, weight).

    Positions mirror merge/srscatter exactly for the bench configuration
    (origin 0, perOut 1/exp, no refine, no jitter): oc+0.5 = (site-2m)*exp
    with m the decoded atlas motion in packed px.
    """
    raw, exp, frames = args.raw, args.exp, args.frames
    if args.jitter == "handheld":
        traj = np.cumsum(rng.standard_normal((frames, 2)) * 0.15 * exp, axis=0)
        traj -= traj.mean(0)
        dxs = 0.5 * exp + traj[:, 0]
        dys = 0.5 * exp + traj[:, 1]
    else:
        dxs = rng.uniform(0, exp, frames)
        dys = rng.uniform(0, exp, frames)
    dxs[0] = 0.0
    dys[0] = 0.0
    i = np.arange(raw)
    X0, Y0 = np.meshgrid(i, i)
    pos, val = [], []
    for f in range(frames):
        dx, dy = float(dxs[f]), float(dys[f])
        # Sampled scene value at the shifted raw-site centers.
        X = X0 * exp - dx
        Y = Y0 * exp - dy
        v = sample_scene(fx, fy, ph, amp, mtf, X, Y, args)
        if args.noise > 0:
            v = v + args.noise * rng.standard_normal((raw, raw))
        # Warp deposited at: site - 2m (packed -> raw is x2), in output
        # pixel-center coordinates. Uniform atlas: m = (dx,dy)/(2*exp).
        mx, my = dx / (2.0 * exp), dy / (2.0 * exp)
        sx = (X0 - 2.0 * mx) * exp
        sy = (Y0 - 2.0 * my) * exp
        pos.append(np.stack([sx.ravel(), sy.ravel()], -1))
        val.append(v.ravel())
    return np.concatenate(pos, 0), np.concatenate(val, 0)


def design(offs, degree):
    """Polynomial basis at sample offsets (ox, oy)."""
    ox, oy = offs[:, 0], offs[:, 1]
    cols = [np.ones_like(ox)]
    if degree >= 1:
        cols += [ox, oy]
    if degree >= 2:
        cols += [ox * ox, oy * oy, ox * oy]
    return np.stack(cols, -1)


def mls_field(args, spos, sval, degree, lam, sw=None):
    """Per-pixel ridge MLS reconstruction on the output grid.

    sw: optional per-sample weight (trust x clip); multiplies the compact
    support window. None = uniform.
    """
    out = args.raw * args.exp
    R = args.radius
    # Grid hash: bin samples into 1px cells, gather (2*ceil(R)+1)^2 cells.
    cell = np.floor(spos).astype(int)
    key = cell[:, 1] * (out + 2) + cell[:, 0]
    order = np.argsort(key)
    skey = key[order]
    H = int(np.ceil(R))
    xs = (np.arange(out) + 0.5)
    field = np.zeros((out, out))
    lam_eff = lam
    for oy in range(out):
        y0 = max(0, oy - H)
        y1 = min(out - 1, oy + H)
        for ox in range(out):
            x0 = max(0, ox - H)
            x1 = min(out - 1, ox + H)
            lo = (y0) * (out + 2) + x0
            hi = (y1) * (out + 2) + x1 + 1
            # cell range is 2D; collect via bounding key range then mask.
            a = np.searchsorted(skey, lo, side="left")
            b = np.searchsorted(skey, hi, side="right")
            idx = order[a:b]
            if len(idx) == 0:
                continue
            p = spos[idx]
            m = (np.abs(p[:, 0] - (ox + 0.5)) <= R) & (np.abs(p[:, 1] - (oy + 0.5)) <= R)
            idx = idx[m]
            if len(idx) == 0:
                continue
            offs = spos[idx] - np.array([ox + 0.5, oy + 0.5])
            A = design(offs, degree)
            v = sval[idx]
            # Bisquare-ish compact support inside R (matches the splat's
            # locality, keeps far samples from ringing the fit).
            d2 = (offs[:, 0] ** 2 + offs[:, 1] ** 2) / (R * R)
            w = np.clip(1.0 - d2, 0.0, 1.0) ** 2 + 1e-6
            if sw is not None:
                w = w * sw[idx]
            W = np.sqrt(w)
            Aw = A * W[:, None]
            vw = v * W
            swsum = w.sum()
            G = Aw.T @ Aw + np.eye(A.shape[1]) * lam_eff * max(swsum, 1e-9)
            try:
                c = np.linalg.solve(G, Aw.T @ vw)
            except np.linalg.LinAlgError:
                c = np.zeros(A.shape[1])
                c[0] = vw.sum() / max(swsum, 1e-9)
            field[oy, ox] = c[0]
    return field


def band_stats(rec, truth_ap, f0=0.25, f1=0.499):
    ref = (slice(8, -8), slice(8, -8))
    x = fft_band(rec, f0, f1)[ref]
    t = fft_band(truth_ap, f0, f1)[ref]
    amp = float(np.sqrt(np.mean(x ** 2)) / max(np.sqrt(np.mean(t ** 2)), 1e-9))
    corr = float(np.corrcoef(x.ravel(), t.ravel())[0, 1])
    return amp, corr


def mls_gated_blend(args, drizzle, spos, sval, smp, degree=2, lam=1e-3):
    """Blend the deposit drizzle with an MLS degree-`degree` fit, gated by
    per-sample motion probability: out = mix(drizzle, mls, 1-max(mp)).
    Where nearby samples disagree coherently (motion/misregistration) the
    output is exactly the shipped drizzle; where they agree it is the
    inverse. No-sample pixels keep the drizzle (the compose falls back).
    Returns (blended, confmean)."""
    out = args.raw * args.exp
    R = args.radius
    cell = np.floor(spos).astype(int)
    key = cell[:, 1] * (out + 2) + cell[:, 0]
    order = np.argsort(key)
    skey = key[order]
    H = int(np.ceil(R))
    field = np.zeros((out, out))
    confsum = 0.0
    for oy in range(out):
        y0 = max(0, oy - H)
        y1 = min(out - 1, oy + H)
        for ox in range(out):
            x0 = max(0, ox - H)
            x1 = min(out - 1, ox + H)
            lo = y0 * (out + 2) + x0
            hi = y1 * (out + 2) + x1 + 1
            a = np.searchsorted(skey, lo, side="left")
            b = np.searchsorted(skey, hi, side="right")
            idx = order[a:b]
            if len(idx) == 0:
                field[oy, ox] = drizzle[oy, ox]
                continue
            p = spos[idx]
            m = (np.abs(p[:, 0] - (ox + 0.5)) <= R) & (np.abs(p[:, 1] - (oy + 0.5)) <= R)
            idx = idx[m]
            if len(idx) == 0:
                field[oy, ox] = drizzle[oy, ox]
                continue
            conf = 1.0 - float(smp[idx].max())
            confsum += conf
            if conf <= 0.0:
                field[oy, ox] = drizzle[oy, ox]
                continue
            offs = spos[idx] - np.array([ox + 0.5, oy + 0.5])
            A = design(offs, degree)
            v = sval[idx]
            d2 = (offs[:, 0] ** 2 + offs[:, 1] ** 2) / (R * R)
            w = np.clip(1.0 - d2, 0.0, 1.0) ** 2 + 1e-6
            W = np.sqrt(w)
            Aw = A * W[:, None]
            vw = v * W
            swsum = w.sum()
            G = Aw.T @ Aw + np.eye(A.shape[1]) * lam * max(swsum, 1e-9)
            try:
                c = np.linalg.solve(G, Aw.T @ vw)
            except np.linalg.LinAlgError:
                field[oy, ox] = drizzle[oy, ox]
                continue
            field[oy, ox] = (1.0 - conf) * drizzle[oy, ox] + conf * c[0]
    return field, confsum / (out * out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=int, default=48)
    ap.add_argument("--exp", type=int, default=2)
    ap.add_argument("--frames", type=int, default=9)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scene", default="tones", choices=["tones", "ramp"])
    ap.add_argument("--jitter", default="uniform", choices=["uniform", "handheld"])
    ap.add_argument("--degrees", default="0,1,2")
    ap.add_argument("--lambdas", default="1e-4,1e-3,1e-2")
    ap.add_argument("--radius", type=float, default=2.5)
    ap.add_argument("--dc", type=float, default=1.2)
    ap.add_argument("--chroma", type=float, default=0.0)
    a = ap.parse_args()

    args = Args()
    for k in ("raw", "exp", "frames", "noise", "seed", "scene", "jitter",
              "dc", "chroma", "radius"):
        setattr(args, k, getattr(a, k))
    rng = np.random.default_rng(a.seed)
    fx, fy, ph, amp, mtf = build_scene(args, rng)
    out = args.raw * args.exp
    spos, sval = collect(args, fx, fy, ph, amp, mtf, rng)
    Xo, Yo = np.meshgrid(np.arange(out) + 0.5, np.arange(out) + 0.5)
    truth_ap = sample_scene(fx, fy, ph, amp, mtf, Xo, Yo, args)
    print("mls_proto: %s exp=%d frames=%d noise=%.3f samples=%d radius=%.1f" % (
        a.scene, a.exp, a.frames, a.noise, len(sval), a.radius))
    for deg in [int(d) for d in a.degrees.split(",")]:
        for lam in [float(v) for v in a.lambdas.split(",")]:
            if deg == 0 and lam != float(a.lambdas.split(",")[0]):
                continue  # degree-0 has no ridge dependence
            rec = mls_field(args, spos, sval, deg, lam)
            amp, corr = band_stats(rec, truth_ap)
            amp_lo, corr_lo = band_stats(rec, truth_ap, 0.25, 0.32)
            amp_hi, corr_hi = band_stats(rec, truth_ap, 0.32, 0.499)
            print("  deg=%d lam=%.0e: [0.25,0.5] amp=%.2fx corr=%.2f | "
                  "lo amp=%.2f corr=%.2f | hi amp=%.2f corr=%.2f" % (
                      deg, lam, amp, corr, amp_lo, corr_lo, amp_hi, corr_hi))
    return 0


if __name__ == "__main__":
    sys.exit(main())
