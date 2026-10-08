#!/usr/bin/env python3
"""Crop replay of a device SrDump through the real drizzle-path shaders.

Loads a crop of the dumped lattice (packed-halves luma+weight), band map
and KernelNet map, renders anisoupscale at 1:1 plus the band deconv, and
saves diagnostics: lattice luma, weight, gate, aniso, final, plus residual
FFTs for grid-pitch identification.

Usage:
    python3 replay_crop.py --cx 4096 --cy 3072 --size 1024 --out /tmp/pix/c
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.normpath(
        os.path.join(HERE, "..", "..", "srsamples", "srdump")))
    ap.add_argument("--cx", type=int, default=4096)
    ap.add_argument("--cy", type=int, default=3072)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--out", default="/tmp/pix/crop")
    args = ap.parse_args()
    import json
    meta = json.load(open(os.path.join(args.dump, "params.json")))

    def rd(name):
        m = meta[name]
        raw = open(os.path.join(args.dump, name + ".raw"), "rb").read()
        a = np.frombuffer(raw, dtype=np.uint32).reshape(m["h"], m["w"])
        return (a.reshape(-1).view(np.float16)
                .reshape(m["h"], m["w"], 2).astype(np.float64), m)

    lat, lm = rd("lattice")
    band, bm = rd("band")
    km = meta["kernels"]
    k = np.frombuffer(open(os.path.join(args.dump, "kernels.raw"), "rb").read(),
                      dtype=np.float16
                      ).reshape(km["rh"], km["rw"], 4).astype(np.float64)
    s = args.size
    x0 = max(0, min(lm["w"] - s, args.cx - s // 2))
    y0 = max(0, min(lm["h"] - s, args.cy - s // 2))
    lum = np.clip(lat[y0:y0 + s, x0:x0 + s, 0], 0, 8)
    wgt = lat[y0:y0 + s, x0:x0 + s, 1]
    bg = band[y0:y0 + s, x0:x0 + s, 0]
    gate = band[y0:y0 + s, x0:x0 + s, 1]
    # Kernels map pitch: lattice 8192 / kernels 1024 = 8 out px per texel.
    mx = x0 // 8
    my = y0 // 8
    ms = s // 8
    kmap = k[my:my + ms, mx:mx + ms]
    if os.environ.get("SR_REPLAY_CONST"):
        kmap = np.zeros_like(kmap)
        kmap[..., 0] = 0.4
        kmap[..., 1] = 0.4
        kmap[..., 3] = 1.0
        print("replay: constant kernel map")
    if os.environ.get("SR_REPLAY_SMOOTH"):
        # Regularize the map spatially (Gaussian, map texels): genuine
        # edge anisotropy is coherent over many texels, per-block
        # directional noise is not.
        from sr_bench import blur as _blur

        sig = float(os.environ["SR_REPLAY_SMOOTH"])
        kmap = np.stack([_blur(kmap[..., c], sig) for c in range(4)], -1)
        print("replay: smoothed kernel map sigma=%.1f" % sig)
    if os.environ.get("SR_REPLAY_SMOOTH_RHO"):
        # Direction-only smoothing (rho channel): keeps sharpness, kills
        # per-block directional noise.
        from sr_bench import blur as _blur

        sig = float(os.environ["SR_REPLAY_SMOOTH_RHO"])
        kmap = kmap.copy()
        kmap[..., 2] = _blur(kmap[..., 2], sig)
        print("replay: rho-only smoothed kernel map sigma=%.1f" % sig)
    print("crop (%d,%d)+%d: luma %.3f..%.3f w %.2f..%.2f gate mean=%.3f" % (
        x0, y0, s, lum.min(), lum.max(), wgt.min(), wgt.max(), gate.mean()))
    print("kernels crop: s1 %.3f..%.3f s2 %.3f..%.3f rho %.3f..%.3f" % (
        kmap[..., 0].min(), kmap[..., 0].max(), kmap[..., 1].min(),
        kmap[..., 1].max(), kmap[..., 2].min(), kmap[..., 2].max()))

    ctx = moderngl.create_standalone_context(backend="egl")
    src = SB.expand_imports(open(os.path.join(
        SB.ASSETS, "upscalecrop/anisoupscale.glsl")).read().lstrip("\n"))
    vsrc = ("#version 310 es\nin vec2 in_pos;\n"
            "void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }\n")
    prog = ctx.program(vertex_shader=vsrc, fragment_shader="#version 310 es\n" + src)
    verts = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4").tobytes())
    try:
        vao = ctx.vertex_array(prog, [(verts, "2f", "in_pos")])
    except Exception:
        vao = ctx.simple_vertex_array(prog, verts, "in_pos")

    def f2(a):
        return np.ascontiguousarray(a.astype(np.float16)).tobytes()

    itex = ctx.texture((s, s), 4, f2(np.stack(
        [lum, lum, lum, np.ones_like(lum)], -1)), dtype="f2")
    mtex = ctx.texture((ms, ms), 4, f2(kmap), dtype="f2")
    tgt = ctx.texture((s, s), 4, dtype="f2")
    fbo = ctx.framebuffer(color_attachments=[tgt])
    prog["InputBuffer"] = 0
    itex.use(0)
    prog["KernelsMap"] = 1
    mtex.use(1)
    prog["fullSize"] = (s, s)
    prog["u_tileOrigin"] = (0, 0)
    prog["u_winOrigin"] = (0, 0)
    prog["u_winFullSize"] = (float(s), float(s))
    prog["sigmaScale"] = 0.55 * 0.45
    prog["sigmaMinPx"] = (0.4, 0.4)
    prog["sigmaMaxPx"] = 1.0
    prog["strength"] = 1.0
    prog["sharpAmt"] = 1.75
    prog["sharpWide"] = 2.2
    prog["acutRel"] = 0.04
    prog["maxElong"] = 8.0
    prog["gateExp"] = 0.3
    prog["srElongCap"] = 1.4
    prog["scaleRatio"] = (1.0, 1.0)
    prog["splitChroma"] = 1
    prog["kernelRadius"] = 5
    prog["debugMode"] = 0
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    vao.render(moderngl.TRIANGLES)
    aniso = np.frombuffer(fbo.read(components=4, dtype="f2"), dtype=np.float16
                          ).reshape(s, s, 4)[..., 0].astype(np.float64)

    # Band deconv (srBandMode=1) with the dumped band map.
    bsrc = SB.expand_imports(open(os.path.join(
        SB.ASSETS, "srpre/band.glsl")).read().lstrip("\n"))
    bprog = ctx.program(vertex_shader=vsrc, fragment_shader="#version 310 es\n" + bsrc)
    try:
        bvao = ctx.vertex_array(bprog, [(verts, "2f", "in_pos")])
    except Exception:
        bvao = ctx.simple_vertex_array(bprog, verts, "in_pos")
    atex = ctx.texture((s, s), 4, f2(np.stack(
        [aniso, aniso, aniso, np.ones_like(aniso)], -1)), dtype="f2")
    h = np.stack([bg, gate], -1).astype(np.float16)
    btex = ctx.texture((s, s), 1, np.ascontiguousarray(h).view(np.uint32).tobytes(),
                       dtype="u4")
    otgt = ctx.texture((s, s), 4, dtype="f2")
    fbo2 = ctx.framebuffer(color_attachments=[otgt])
    bprog["InputBuffer"] = 0
    atex.use(0)
    bprog["BandMap"] = 1
    btex.use(1)
    bprog["srPerOut"] = (0.5, 0.5)
    bprog["srBlack"] = (0.0, 0.0, 0.0)
    bprog["u_inOrigin"] = (0, 0)
    bprog["u_tileOrigin"] = (0, 0)
    bprog["srBandMode"] = 1.0
    fbo2.use()
    fbo2.clear(0.0, 0.0, 0.0, 1.0)
    bvao.render(moderngl.TRIANGLES)
    fin = np.frombuffer(fbo2.read(components=4, dtype="f2"), dtype=np.float16
                        ).reshape(s, s, 4)[..., 0].astype(np.float64)

    from PIL import Image
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def save(path, a):
        a = a - a.min()
        a = a / max(a.max(), 1e-9) * 255.0
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(path)

    save(args.out + "_luma.png", lum)
    save(args.out + "_weight.png", wgt)
    save_png_gate = args.out + "_gate.png"
    save(save_png_gate, gate)
    save(args.out + "_aniso.png", aniso)
    save(args.out + "_final.png", fin)
    resid = fin - aniso
    save(args.out + "_banddelta.png", resid * 8.0 + 0.5)
    if os.environ.get("SR_REPLAY_NPY"):
        np.save(args.out + "_lum.npy", lum)
        np.save(args.out + "_aniso.npy", aniso)
        np.save(args.out + "_fin.npy", fin)
        print("replay: saved float arrays")

    # Residual pitch ID: radial-band energies of the aniso HF + the
    # band-applied delta, and sample/map-pitch autocorrelation of each.
    def pitch(a, tag):
        a = a - a.mean()
        N = a.shape[0]
        F = np.abs(np.fft.fftshift(np.fft.fft2(a * np.outer(np.hanning(N), np.hanning(N))))) ** 2
        h, w = F.shape
        cy, cx = h // 2, w // 2
        Y, X = np.mgrid[0:h, 0:w]
        r = np.sqrt(((Y - cy) / h) ** 2 + ((X - cx) / w) ** 2)
        low = (r > 0.03) & (r < 0.10)
        out = {}
        for f0, nm in ((0.5, "nyq"), (0.25, "half"), (0.125, "map")):
            bd = (r > f0 - 0.03) & (r < f0 + 0.03)
            out[nm] = float(F[bd].mean() / max(F[low].mean(), 1e-9))
        den = max(float((a ** 2).mean()), 1e-12)
        ac = 0.0
        for dy, dx in ((0, 2), (2, 0), (2, 2)):
            b = np.roll(np.roll(a, dy, 0), dx, 1)
            ac += float((a * b).mean()) / den
        out["ac"] = ac / 3.0
        print("  pitch %-6s nyq=%.2f half=%.2f map=%.2f ac=%+.2f" % (tag, out["nyq"], out["half"], out["map"], out["ac"]))
        return out

    m = (slice(64, -64), slice(64, -64))
    pitch(lum[m], "lattice")
    pitch(aniso[m], "aniso")
    pitch(fin[m], "final")
    pitch((fin - aniso)[m], "delta")
    return 0


if __name__ == "__main__":
    sys.exit(main())
