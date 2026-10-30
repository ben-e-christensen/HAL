#!/usr/bin/env python3
"""
How fast can the HAL array drag a bead around before losing it?

Keep this file next to hal_levsim.py.

What it does
------------
1. Real dynamics (not the quasi-static GUI motion): the bead has mass and is
   pushed by the Gor'kov force + gravity + air drag, integrated in time
   (100 us steps). The trap moves along an outward Archimedean spiral at a
   chosen speed; phases are re-solved every `update` seconds (100 us ~ the
   on-FPGA engine; ~11 ms ~ stock "PC streams 256 phases over serial").
2. Finds the max speed by doubling the speed until the bead is lost, then
   bisecting between the last good and first bad speed.
3. Renders a REAL-TIME GIF of the spiral at a safe fraction of that speed
   (50 fps; background = time-averaged |p|, no 40 kHz flicker; bead + trail),
   plus a 10x slow-motion copy, since at a few m/s the bead covers a whole
   turn of the spiral between real-time frames.

Run:
    python speed_test.py                          # all three arrangements, FPGA update rate
    python speed_test.py --setups opposed --compare-serial
    python speed_test.py --setups free --z 50 --fraction 0.8 --vpp 12
    python speed_test.py --speed 0.5 --setups opposed   # skip search, just make the GIF at 0.5 m/s

Model limits (on top of hal_levsim's): drag uses a standard sphere drag
curve in still air; acoustic streaming (air flow driven by the sound) and
bead rotation are ignored; the bead does not change the field. Treat the
speeds as model estimates to compare setups and update rates, not as
guaranteed lab numbers.
"""

import argparse
import math
import os
import time

import numpy as np
import hal_lev_sim as h

DT = 1e-4                 # integration step, s
ETA_AIR = 1.8e-5          # air viscosity, Pa s
LOST_DIST = 0.015         # bead counted as lost if > 15 mm from the trap target
HOLD_AFTER = 0.15         # s to hold at the end before judging "held"
FINAL_DIST = 0.010        # must be within 10 mm of the target after the hold
RAMP = 0.05               # s, trap ramps 0 -> v (and v -> 0 at the end) over this time

SETUP_ALIASES = {"opposed": "two opposed boards", "reflector": "single + reflector",
                 "free": "single, free field"}
DEFAULT_Z = 80.0          # same spiral height for every arrangement, mm


# ----------------------------------------------------------------------------
# Path: Archimedean spiral parameterised by arc length
# ----------------------------------------------------------------------------
class Spiral:
    def __init__(self, z_mm, r0_mm=5.0, r1_mm=60.0, pitch_mm=10.0, cx=0.0, cy=0.0):
        r0, r1, pitch = r0_mm / 1e3, r1_mm / 1e3, pitch_mm / 1e3
        th1 = (r1 - r0) / pitch * 2 * math.pi
        th = np.linspace(0, th1, 20000)
        r = r0 + pitch * th / (2 * math.pi)
        self.x = cx + r * np.cos(th)
        self.y = cy + r * np.sin(th)
        self.z = z_mm / 1e3
        seg = np.hypot(np.diff(self.x), np.diff(self.y))
        self.s = np.concatenate([[0], np.cumsum(seg)])
        self.length = self.s[-1]
        self.r0, self.r1 = r0, r1

    def pos(self, s):
        s = min(max(s, 0.0), self.length)
        return np.array([np.interp(s, self.s, self.x), np.interp(s, self.s, self.y), self.z])


# ----------------------------------------------------------------------------
# Dynamics
# ----------------------------------------------------------------------------
def drag(bead, v):
    """Sphere drag in still air (Schiller-Naumann, valid to Re ~ 800)."""
    sp = np.linalg.norm(v)
    if sp == 0:
        return np.zeros(3)
    d = 2 * bead.r
    re = h.RHO_AIR * sp * d / ETA_AIR
    cd = 24 / re * (1 + 0.15 * re ** 0.687)
    area = math.pi * bead.r ** 2
    return -0.5 * h.RHO_AIR * cd * area * sp * v


def make_sim(setup, vpp, bead_mm, gap_mm, quantize=True):
    s = h.Sim()
    s.setup = setup
    s.signature = h.RECOMMENDED_SIGNATURE[setup]
    s.vpp, s.bd, s.H, s.quantize = vpp, h.Bead(bead_mm / 1e3), gap_mm / 1e3, quantize
    return s


def run(sim, path, speed, update=DT, record_every=None, trail_every=5e-4, verbose=False):
    """
    Drag the trap along `path` at `speed` (m/s). Returns dict with
    held (bool), t_lost, and optionally recorded frames.
    """
    sim.target = path.pos(0.0)
    sim.rebuild()
    sim.reset_bead()                      # settle into the starting trap
    if sim.dropped:
        return {"held": False, "t_lost": 0.0, "frames": [], "reason": "no trap at start"}
    x = sim.bead.copy()
    v = np.zeros(3)
    m = sim.bd.m
    g = np.array([0, 0, -h.G * m]) if sim.gravity else np.zeros(3)
    s_path, t, t_next_update = 0.0, 0.0, 0.0
    t_end_path = None
    frames, t_next_rec, t_next_trail = [], 0.0, 0.0
    trail = []
    z0, z1 = sim.z_limits()
    while True:
        # trap command (stepped at the update rate, like the hardware)
        # trapezoidal speed profile: ramp up over RAMP, brake to a stop at the end
        acc = speed / RAMP
        rem = max(path.length - s_path, 0.0)
        sp = min(speed, acc * t, math.sqrt(2 * acc * rem))
        s_path += max(sp, 1e-6) * DT
        if t >= t_next_update:
            sim.target = path.pos(s_path)
            sim.rebuild()
            t_next_update += update
        # forces: acoustic (Gor'kov, no gravity term) + gravity + drag
        F = h.force(sim.boards, x, sim.p0, sim.bd, gravity=False) + g + drag(sim.bd, v)
        v = v + DT * F / m
        x = x + DT * v
        t += DT
        if record_every is not None and t >= t_next_trail:
            trail.append(x.copy())
            t_next_trail += trail_every
        if record_every is not None and t >= t_next_rec:
            frames.append({"t": t, "bead": x.copy(), "target": sim.target.copy(),
                           "trail": np.array(trail)})
            t_next_rec += record_every
        dist = np.linalg.norm(x - sim.target)
        if dist > LOST_DIST or x[2] < z0 + sim.bd.r or x[2] > z1 - sim.bd.r:
            t_trigger = t
            if record_every is not None:   # keep simulating so the GIF shows the fall
                for _ in range(int(0.4 / DT)):
                    F = h.force(sim.boards, x, sim.p0, sim.bd, gravity=False) + g + drag(sim.bd, v)
                    v = v + DT * F / m
                    x = x + DT * v
                    t += DT
                    if x[2] <= z0 + sim.bd.r:          # landed on the reflector / bottom board
                        x[2] = z0 + sim.bd.r
                        frames.append({"t": t, "bead": x.copy(), "target": sim.target.copy(),
                                       "trail": np.array(trail + [x.copy()]), "lost": True})
                        break
                    if t >= t_next_trail:
                        trail.append(x.copy())
                        t_next_trail += trail_every
                    if t >= t_next_rec:
                        frames.append({"t": t, "bead": x.copy(), "target": sim.target.copy(),
                                       "trail": np.array(trail), "lost": True})
                        t_next_rec += record_every
            return {"held": False, "t_lost": t_trigger, "frames": frames,
                    "reason": f"lost at s={s_path*1e3:.0f} mm"}
        if s_path >= path.length - 1e-6 and t_end_path is None:
            t_end_path = t
        if t_end_path is not None and t - t_end_path >= HOLD_AFTER:
            ok = np.linalg.norm(x - sim.target) < FINAL_DIST
            return {"held": ok, "t_lost": None, "frames": frames, "reason": "ok" if ok else "drifted off"}


def find_max_speed(setup, path, update, vpp, bead_mm, gap_mm, v_start=0.25, v_cap=20.0,
                   rel_tol=0.05, log=print):
    """Double until the bead is lost, then bisect. Returns (v_max_held, v_min_lost)."""
    lo, hi, v = 0.0, None, v_start
    while hi is None:
        t0 = time.time()
        r = run(make_sim(setup, vpp, bead_mm, gap_mm), path, v, update)
        log(f"    {v:6.3f} m/s -> {'held' if r['held'] else 'LOST'} ({r['reason']}, {time.time()-t0:.0f}s)")
        if r["held"]:
            lo, v = v, v * 2
            if v > v_cap:
                return lo, None
        else:
            if lo == 0.0 and v <= v_start and r["reason"] == "no trap at start":
                return 0.0, 0.0
            hi = v
    while (hi - lo) / hi > rel_tol and lo > 0:
        v = 0.5 * (lo + hi)
        t0 = time.time()
        r = run(make_sim(setup, vpp, bead_mm, gap_mm), path, v, update)
        log(f"    {v:6.3f} m/s -> {'held' if r['held'] else 'LOST'} ({r['reason']}, {time.time()-t0:.0f}s)")
        lo, hi = (v, hi) if r["held"] else (lo, v)
    if lo == 0:   # fails even at v_start: search downward
        v = v_start
        while v > 0.03:   # below this the path itself has no trap somewhere; not a speed limit
            v /= 2
            r = run(make_sim(setup, vpp, bead_mm, gap_mm), path, v, update)
            log(f"    {v:6.3f} m/s -> {'held' if r['held'] else 'LOST'} ({r['reason']})")
            if r["held"]:
                return v, 2 * v
        return 0.0, v
    return lo, hi


# ----------------------------------------------------------------------------
# Real-time GIF
# ----------------------------------------------------------------------------
def make_gif(setup, path, speed, update, vpp, bead_mm, gap_mm, fn, fps=50, grid=80, label="",
             slowmo=1.0, window=None):
    """
    slowmo = 1 -> real time; 10 -> 10x slow motion.
    window = seconds before the bead is lost to start the GIF (None = whole run);
             used for slow-motion clips of a loss so they stay short.
    Left: top view at the spiral height. Right: side view (x-z) through the
    trap, so a fall is visible.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.animation import PillowWriter

    sim = make_sim(setup, vpp, bead_mm, gap_mm)
    r = run(sim, path, speed, update, record_every=1.0 / (fps * slowmo))
    frames = r["frames"]
    if window is not None and r["t_lost"]:
        frames = [f for f in frames if f["t"] >= r["t_lost"] - window]
    ext = 0.075
    z0, z1 = sim.z_limits()
    xs = np.linspace(-ext, ext, grid)
    X, Y = np.meshgrid(xs, xs)
    top_pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, path.z)])
    zs = np.linspace(z0 + 1e-3, z1 - 1e-3, grid)
    XS, ZS = np.meshgrid(xs, zs)

    def side_pts(y):
        return np.column_stack([XS.ravel(), np.full(XS.size, y), ZS.ravel()])

    mm = 1e3
    fig = Figure(figsize=(11, 5.0), dpi=90, layout="constrained")
    ax, axs = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    sim.target = path.pos(0.0); sim.rebuild()
    p0, _ = h.field(sim.boards, top_pts, sim.p0, want_grad=False)
    vmax = np.abs(p0).max() / 1000 * 1.05          # fixed colour scale for the whole GIF
    kw = dict(origin="lower", cmap="inferno", vmin=0, vmax=vmax, interpolation="bilinear")
    im = ax.imshow(np.abs(p0).reshape(X.shape) / 1000, extent=[-ext * mm, ext * mm] * 2, **kw)
    ims = axs.imshow(np.zeros(XS.shape), extent=[-ext * mm, ext * mm, zs[0] * mm, zs[-1] * mm],
                     aspect="equal", **kw)
    cb = fig.colorbar(im, ax=[ax, axs], shrink=0.8, pad=0.02)
    cb.set_label("|p| (kPa), time-averaged")
    ax.plot(path.x * mm, path.y * mm, color="w", lw=0.6, alpha=0.35, ls="--")
    trail_ln, = ax.plot([], [], color="#58c4dd", lw=1.5)
    tgt_mk, = ax.plot([], [], "x", color="w", ms=9, mew=2, path_effects=h._OUTLINE)
    bead_mk, = ax.plot([], [], "o", mfc="w", mec="k", ms=8)
    for _, zb in sim.real_boards:
        axs.plot([-ext * mm, ext * mm], [zb * mm] * 2, color="#58c4dd", lw=4, solid_capstyle="butt")
    if setup == "single + reflector":
        axs.plot([-ext * mm, ext * mm], [0.5, 0.5], color="0.75", lw=3)
    axs.axhline(path.z * mm, color="w", lw=0.6, alpha=0.35, ls="--")
    s_trail, = axs.plot([], [], color="#58c4dd", lw=1.5)
    s_tgt, = axs.plot([], [], "x", color="w", ms=9, mew=2, path_effects=h._OUTLINE)
    s_bead, = axs.plot([], [], "o", mfc="w", mec="k", ms=8)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=9,
                  color="w", family="monospace")
    lost_txt = axs.text(0.5, 0.92, "", transform=axs.transAxes, ha="center", va="top",
                        fontsize=13, fontweight="bold", color="#ff5a5a")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    axs.set_xlabel("x (mm)"); axs.set_ylabel("z (mm)")
    tag = "real time" if slowmo == 1 else f"{slowmo:g}x slow motion"
    ax.set_title(f"Top view at z = {path.z*mm:.0f} mm", fontsize=10)
    axs.set_title("Side view (x-z) through the trap", fontsize=10)
    fig.suptitle(f"{setup}, outward spiral  ({tag})", fontsize=11)

    writer = PillowWriter(fps=fps)
    with writer.saving(fig, fn, 70):               # 70 dpi keeps long GIFs to a few MB
        for f in frames:
            sim.target = f["target"]; sim.rebuild()
            p, _ = h.field(sim.boards, top_pts, sim.p0, want_grad=False)
            im.set_data(np.abs(p).reshape(X.shape) / 1000)
            ps, _ = h.field(sim.boards, side_pts(f["target"][1]), sim.p0, want_grad=False)
            ims.set_data(np.abs(ps).reshape(XS.shape) / 1000)
            tr = f["trail"]
            trail_ln.set_data(tr[:, 0] * mm, tr[:, 1] * mm)
            recent = tr[-200:]                     # last ~0.1 s in the side view
            s_trail.set_data(recent[:, 0] * mm, recent[:, 2] * mm)
            tgt_mk.set_data([f["target"][0] * mm], [f["target"][1] * mm])
            bead_mk.set_data([f["bead"][0] * mm], [f["bead"][1] * mm])
            s_tgt.set_data([f["target"][0] * mm], [f["target"][2] * mm])
            s_bead.set_data([f["bead"][0] * mm], [f["bead"][2] * mm])
            txt.set_text(f"t = {f['t']*1e3:6.0f} ms\nv = {speed:.3f} m/s {label}")
            lost_txt.set_text("BEAD LOST" if f.get("lost") else "")
            writer.grab_frame()
        for _ in range(max(fps // 2, 1)):          # short pause before the loop restarts
            writer.grab_frame()
    return r


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setups", nargs="+", choices=list(SETUP_ALIASES), default=["opposed", "reflector", "free"],
                    help="arrangements to test (default: all three)")
    ap.add_argument("--z", type=float, help="spiral height in mm (default 80, all arrangements)")
    ap.add_argument("--r0", type=float, default=5.0, help="spiral start radius, mm")
    ap.add_argument("--r1", type=float, default=60.0, help="spiral end radius, mm")
    ap.add_argument("--pitch", type=float, default=10.0, help="radius gained per turn, mm")
    ap.add_argument("--vpp", type=float, default=12.0)
    ap.add_argument("--bead", type=float, default=2.0, help="bead diameter, mm")
    ap.add_argument("--gap", type=float, default=160.0)
    ap.add_argument("--update-us", type=float, default=100.0,
                    help="trap update interval in microseconds (default 100 ~ on-FPGA engine)")
    ap.add_argument("--compare-serial", action="store_true",
                    help="also find the max speed with ~11 ms updates (stock serial streaming)")
    ap.add_argument("--fraction", type=float, default=0.85, help="GIF speed as a fraction of the max")
    ap.add_argument("--speed", type=float, help="skip the search; make the GIF at this speed (m/s)")
    ap.add_argument("--slowmo", type=float, default=10.0,
                    help="also save a slow-motion GIF at this factor (0 = skip; default 10)")
    ap.add_argument("--out", default="speed_output")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rows = []
    for key in a.setups:
        setup = SETUP_ALIASES[key]
        z = a.z if a.z is not None else DEFAULT_Z
        path = Spiral(z, a.r0, a.r1, a.pitch)
        print(f"\n== {setup}: spiral r {a.r0:g}->{a.r1:g} mm, pitch {a.pitch:g} mm, "
              f"z = {z:g} mm, path length {path.length*1e3:.0f} mm")
        upd = a.update_us * 1e-6
        if a.speed is None:
            print(f"  searching max speed, trap update every {a.update_us:g} us")
            lo, hi = find_max_speed(setup, path, upd, a.vpp, a.bead, a.gap)
            print(f"  -> max speed held: {lo:.3f} m/s (lost at {hi if hi else '>20'} m/s)")
            rows.append((setup, z, a.update_us, lo, hi))
            if a.compare_serial:
                print("  searching max speed, trap update every 11 ms (stock serial)")
                lo_s, hi_s = find_max_speed(setup, path, 11e-3, a.vpp, a.bead, a.gap, v_start=0.05)
                print(f"  -> max speed held: {lo_s:.3f} m/s")
                rows.append((setup, z, 11000.0, lo_s, hi_s))
            if lo > 0:
                v_gif = a.fraction * lo
                label = f"({a.fraction:.0%} of max)"
            else:                       # lost at every speed: show it at the slowest one tried
                v_gif = hi or 0.0
                label = "(slowest tried, bead lost)"
        else:
            v_gif, label = a.speed, ""
        if v_gif <= 0:
            print("  no trap even at the start of the spiral; skipping GIF (try another --z, --vpp or --bead)")
            continue
        fn = os.path.join(a.out, f"spiral_{h.SETUP_FOLDERS[setup]}_{v_gif:.3f}mps.gif")
        t0 = time.time()
        r = make_gif(setup, path, v_gif, upd, a.vpp, a.bead, a.gap, fn, label=label)
        print(f"  GIF at {v_gif:.2f} m/s: {fn}  ({'held' if r['held'] else 'LOST: ' + r['reason']}, "
              f"{time.time()-t0:.0f}s to render)")
        if a.slowmo and a.slowmo > 1:
            fn2 = fn.replace(".gif", f"_slowmo{a.slowmo:g}x.gif")
            # for a loss, only the last 0.3 s before it (+ the fall), so the clip stays short
            make_gif(setup, path, v_gif, upd, a.vpp, a.bead, a.gap, fn2, fps=25, label=label,
                     slowmo=a.slowmo, window=None if r["held"] else 0.3)
            print(f"  slow-motion GIF: {fn2}")

    if rows:
        with open(os.path.join(a.out, "max_speeds.csv"), "w") as f:
            f.write("arrangement,z_mm,update_us,max_speed_held_mps,first_lost_mps\n")
            for s, z, u, lo, hi in rows:
                f.write(f"{s},{z:g},{u:g},{lo:.4f},{'' if hi is None else f'{hi:.4f}'}\n")
        print(f"\nsummary written to {os.path.join(a.out, 'max_speeds.csv')}")


if __name__ == "__main__":
    main()