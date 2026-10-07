#!/usr/bin/env python3
"""Motion-gate bench for merge/srscatter.glsl: proves the absolute motion gate
(SR_MOTION_K1/K2) with the REAL shader on a synthetic constant-residual field.

Setup (per site, all frames): a constant luma field, nonzero atlas motion
(so the mock LK warp would apply), a uniform mock srRefMap (.xy = correction,
.z = tiny post-correction residual, neighbour divergence zero by uniformity),
and packed (diff, base) trust textures separated by a CONSTANT offset. A
constant offset survives the 3x3 signed mean, so preRes is known exactly and
the gate decision is deterministic:

- offset 0.2, band 0.05 -> preRes=0.2 > 2*band -> motionProb=1:
  warp suppressed AND lf=preRes (low weight). Pre-fix code trusted refv.z
  (weight ~1, warped) - the moving-subject failure.
- offset 0 -> motionProb=0: warp applies, weight ~1 (no false gating).

Usage:
    python3 motion_gate_bench.py            # PASS/FAIL + exit code
    python3 motion_gate_bench.py --k1 10 --k2 20   # gate disabled -> weights stay ~1
"""
import argparse
import os
import sys

import numpy as np
import moderngl

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import sr_bench as SB  # noqa: E402


def f2(a):
    return np.ascontiguousarray(a.astype(np.float16)).tobytes()


def run_case(ctx, raw, out, exp, frames, deltas, offset, ref_corr, ref_resid,
             k1, k2, band=0.05):
    """One deposit run. Returns (meanW_interior, depv)."""
    asz = max(1, raw // 16)
    neutral = np.full((raw // 2, raw // 2, 4), 0.5, dtype=np.float16)
    base_tex = ctx.texture((raw // 2, raw // 2), 4, f2(neutral), dtype="f2")
    atlas = ctx.texture((asz, asz), 4, dtype="f2")
    mock = np.zeros((asz, asz, 4), dtype=np.float16)
    mock[..., 0] = ref_corr[0]
    mock[..., 1] = ref_corr[1]
    mock[..., 2] = ref_resid
    mock[..., 3] = 1.0
    ref_tex = ctx.texture((asz, asz), 4, f2(mock), dtype="f2")
    # Trust pair with an exact constant offset (base 0.5, diff 0.5+offset).
    barr = np.full((raw // 2, raw // 2, 4), 0.5, dtype=np.float16)
    darr = np.full((raw // 2, raw // 2, 4), 0.5 + offset, dtype=np.float16)
    diff_tex = ctx.texture((raw // 2, raw // 2), 4, f2(darr), dtype="f2")
    base_scene = ctx.texture((raw // 2, raw // 2), 4, f2(barr), dtype="f2")
    # Constant luma field (luma=own=1.0: clip weight 1, expose 1).
    luma = np.zeros((raw, raw, 4), dtype=np.float16)
    luma[..., 0] = 1.0
    luma[..., 1] = 1.0
    depv = ctx.texture((out, out), 1, dtype="u4")
    depw = ctx.texture((out, out), 1, dtype="u4")
    depv.bind_to_image(0, read=True, write=True)
    depw.bind_to_image(1, read=True, write=True)
    clear = ctx.compute_shader(SB.load_compute("merge/srclear.glsl"))
    clear["srClearA"] = 0
    clear["srClearB"] = 1
    clear.run(out, out, 1)
    defines = {"SR_MOTION_K1": "%.4f" % k1, "SR_MOTION_K2": "%.4f" % k2}
    try:
        sc = ctx.compute_shader(SB.load_compute(
            "merge/srscatter.glsl", defines=defines))
    except Exception:
        sc = ctx.compute_shader(SB.load_compute(
            "merge/srscatter.glsl", version="#version 430 core",
            drop_oes=True, defines=defines))
    sc["srLumaTex"] = 0
    sc["diffPacked"] = 2
    sc["basePacked"] = 3
    sc["alignmentTexture"] = 4
    sc["srRefMap"] = 5
    diff_tex.use(2)
    base_scene.use(3)
    ref_tex.use(5)
    sc["srDepV"] = 0
    sc["srDepW"] = 1
    sc["srRefine"] = 1.0
    sc["srFlowAlign"] = 0
    sc["srShift"] = (0, 0)
    sc["srAlignSize"] = (asz, asz)
    sc["srRawHalf"] = (raw // 2, raw // 2)
    sc["srCfa"] = (0, 0)
    sc["srFullPerOut"] = (1.0 / exp, 1.0 / exp)
    sc["srOrigin"] = (0.0, 0.0)
    sc["srExpose"] = 1.0
    sc["srZeroMotion"] = 0.0
    sc["srJitter"] = 0.0
    sc["srMotionMax"] = 64.0
    sc["srTrustFloor"] = 0.0
    sc["srTrustBand"] = band
    sc["srNoiseS0"] = 0.0
    sc["srNoiseO0"] = 0.0
    sc["srClipAtten"] = 0.0
    mot = np.zeros((asz, asz, 4), dtype=np.float16)
    ltex = ctx.texture((raw, raw), 4, f2(luma), dtype="f2")
    for f in range(frames):
        dx, dy = deltas[f]
        mot[..., 2] = dx / (2.0 * exp)
        mot[..., 3] = dy / (2.0 * exp)
        atlas.write(f2(mot))
        ltex.use(0)
        atlas.use(4)
        sc["srFrame"] = f
        sc.run(raw, raw, 1)
    gv = np.frombuffer(depv.read(), dtype=np.uint32
                       ).reshape(out, out).astype(np.float64) / SB.FIXED
    gw = np.frombuffer(depw.read(), dtype=np.uint32
                       ).reshape(out, out).astype(np.float64) / SB.FIXED
    for t in (base_tex, atlas, ref_tex, diff_tex, base_scene, ltex,
              depv, depw):
        t.release()
    m = (slice(4, -4), slice(4, -4))
    return float(gw[m].mean()), gv[m].copy(), gw[m].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=int, default=32)
    ap.add_argument("--exp", type=int, default=2)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--k1", type=float, default=1.0)
    ap.add_argument("--k2", type=float, default=2.0)
    args = ap.parse_args()
    raw, exp, frames = args.raw, args.exp, args.frames
    out = raw * exp
    # Nonzero per-frame motion so the mock warp has authority
    # (clamp(2|m|) > 0); constant luma keeps the geometry deterministic.
    deltas = [(0.0, 0.0)] + [(1.0, 0.5)] * (frames - 1)

    ctx = moderngl.create_standalone_context(backend="egl")
    ok = True

    def check(name, cond, detail=""):
        nonlocal_ok = ok
        print("  gate: %-52s %s %s" % (name, "ok" if cond else "FAIL", detail))
        return nonlocal_ok and cond

    # 1. Gate fires on a large coherent residual: weight collapses.
    w_motion, v_motion, _ = run_case(
        ctx, raw, out, exp, frames, deltas, offset=0.2,
        ref_corr=(0.3, 0.1), ref_resid=0.001, k1=args.k1, k2=args.k2)
    # w = 1/(1+(0.2/0.05)^2)/1 = 1/17 ~= 0.059 per site (mag term ~1).
    ok = check("motion residual down-weighted", w_motion < 0.15,
               "(meanW=%.4f, expect ~0.06)" % w_motion)

    # 2. No false fire on a clean frame: full weight.
    w_clean, v_clean, _ = run_case(
        ctx, raw, out, exp, frames, deltas, offset=0.0,
        ref_corr=(0.3, 0.1), ref_resid=0.001, k1=args.k1, k2=args.k2)
    # Each site deposits total weight 1 -> meanW = frames/exp^2.
    expect = frames / float(exp * exp)
    ok = check("clean frame keeps weight", abs(w_clean - expect) < 0.1,
               "(meanW=%.4f, expect %.2f)" % (w_clean, expect))

    # 3. Warp suppressed on motion: corrected deposit == uncorrected deposit.
    _, v_nocorr, _ = run_case(
        ctx, raw, out, exp, frames, deltas, offset=0.2,
        ref_corr=(0.0, 0.0), ref_resid=0.001, k1=args.k1, k2=args.k2)
    warp_diff = float(np.abs(v_motion - v_nocorr).max())
    ok = check("warp suppressed on motion", warp_diff == 0.0,
               "(max|d|=%.6f)" % warp_diff)

    # 4. Warp applies without motion: the mock correction moves deposits.
    warp_clean = float(np.abs(v_clean - run_case(
        ctx, raw, out, exp, frames, deltas, offset=0.0,
        ref_corr=(0.0, 0.0), ref_resid=0.001,
        k1=args.k1, k2=args.k2)[1]).max())
    ok = check("warp applies when clean", warp_clean > 0.0,
               "(max|d|=%.4f)" % warp_clean)

    print("  %s (k1=%.2f k2=%.2f)" % ("PASS" if ok else "FAIL",
                                      args.k1, args.k2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
