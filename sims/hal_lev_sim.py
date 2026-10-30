#!/usr/bin/env python3
"""
HAL levitation simulator
========================

Linear acoustic field + Gor'kov trap simulation of the SonicSurface
16x16, 40 kHz phased array, with a GUI where a simulated bead follows the
trap as you send it new (x, y, z) coordinates.

Setups
------
  1. single + reflector : one board facing down at height H over a flat
                          reflector at z = 0 (the UpnaLab single-board setup)
  2. single, free field : one board facing up at z = 0, nothing above it.
                          Needs a twin-trap or vortex signature to hold a bead.
  3. two opposed boards : bottom board at z = 0 facing up, top board at z = H
                          facing down.

Coordinates are in mm. x, y = 0 is the centre of the array; z is height
above the reflector / bottom board (setup 3) / board face (setup 2).

What the model predicts (2 mm EPS bead, 12 Vpp, 5-bit phases):
  * two opposed boards : strong traps; a plain focus traps at the standing-
                         wave node nearest the target (nodes ~4.3 mm apart).
  * single, free field : twin trap / vortex hold, sagging a few mm under
                         gravity; traps weaken with height (z ~ 90 mm fails).
  * single + reflector : bead height locks to the standing-wave nodes set by
                         the reflector, so z commands only hop between nodes;
                         a plain focus pushes the bead onto a ring beside it.
  Try the Drive and Bead size fields to see where traps appear or fail.

Requirements:  Python 3.9+, numpy, matplotlib >= 3.6, tkinter (ships with the
standard python.org installers on Windows/macOS; on Linux install python3-tk).

Run:           python hal_levsim.py
Headless test: python hal_levsim.py --selftest   (writes selftest.svg)

Model notes (what this is NOT):
  * Linear superposition of ideal piston sources (no nonlinearity, no
    scattering from the bead, no mutual coupling between transducers).
  * Reflector = perfect rigid mirror (image-source method).
  * Gor'kov potential (valid for beads much smaller than the wavelength;
    a 2 mm bead at lambda = 8.6 mm is at the edge of that).
  * Transducer output P0 is an approximate datasheet-style value; the
    drivers output a square wave, so the 40 kHz fundamental is 4/pi times
    the square-wave Vpp.
  * The bead moves quasi-statically: each frame it slides downhill on the
    trap potential (incl. gravity) at a capped speed. That shows where it
    ends up and whether it is held, not its real dynamics (a real bead
    oscillates in the trap at tens to hundreds of Hz).
"""

import math
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------
# Physical constants and array geometry
# ----------------------------------------------------------------------------
F_HZ = 40e3                 # drive frequency
C_AIR = 343.0               # speed of sound in air, m/s (~20 C)
RHO_AIR = 1.2               # kg/m^3
G = 9.81

N_SIDE = 16                 # 16 x 16 emitters
PITCH = 10.5e-3             # m, from the HAL pick-and-place file (413.4 mil)
PISTON_RADIUS = 4.5e-3      # m, effective radius of a 10 mm transducer
P0_PER_VOLT = 0.17          # Pa*m per Vpp at 1 m (approximate, MSO-P1040H07T class)

# Bead: expanded polystyrene by default
BEAD_RHO = 29.0             # kg/m^3
BEAD_C = 900.0              # m/s

OMEGA = 2 * math.pi * F_HZ
K = OMEGA / C_AIR
LAMBDA = C_AIR / F_HZ

# Emitter positions in the board plane, raster order (row-major).
_idx = (np.arange(N_SIDE) - (N_SIDE - 1) / 2) * PITCH
EMX, EMY = np.meshgrid(_idx, _idx, indexing="xy")
EMX = EMX.ravel()
EMY = EMY.ravel()
N_EM = EMX.size


# ----------------------------------------------------------------------------
# Piston directivity: 2 J1(x) / x, with J1 from its power series
# (k*a*sin(theta) <= 3.3 here, so 20 terms is far more than enough;
#  avoids a scipy dependency)
# ----------------------------------------------------------------------------
_J1_COEF = np.array([(-1) ** m / (math.factorial(m) * math.factorial(m + 1) * 2 ** (2 * m + 1))
                     for m in range(20)])


def piston_directivity(sin_theta):
    x = K * PISTON_RADIUS * sin_theta
    x2 = x * x
    # 2*J1(x)/x = 2 * sum c_m x^(2m)
    acc = np.zeros_like(x)
    for c in _J1_COEF[::-1]:
        acc = acc * x2 + c
    return 2.0 * acc


# ----------------------------------------------------------------------------
# Sources: a list of boards. Each board = (z position, normal sign, phases)
# normal sign +1 faces +z (up), -1 faces -z (down).
# ----------------------------------------------------------------------------
class Board:
    def __init__(self, z, normal, phases, amp_scale=1.0, enabled=None):
        self.z = z
        self.normal = normal
        self.phases = np.asarray(phases, dtype=float)
        self.amp = amp_scale
        self.enabled = np.ones(N_EM, bool) if enabled is None else enabled


def field(boards, pts, p0, want_grad=True):
    """
    Complex pressure (Pa) and its gradient at points pts (N,3), metres.
    Returns p (N,), grad (N,3). Gradient uses the far-field monopole
    derivative s*(ik - 1/r)*r_hat and ignores the directivity derivative.
    Processes points in chunks to bound memory.
    """
    pts = np.atleast_2d(pts)
    n = pts.shape[0]
    p = np.zeros(n, complex)
    g = np.zeros((n, 3), complex) if want_grad else None
    chunk = max(1, 400_000 // N_EM)
    for b in boards:
        ex, ey = EMX[b.enabled], EMY[b.enabled]
        ph = b.phases[b.enabled]
        for s in range(0, n, chunk):
            P = pts[s:s + chunk]
            dx = P[:, 0:1] - ex[None, :]
            dy = P[:, 1:2] - ey[None, :]
            dz = P[:, 2:3] - b.z
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            r = np.maximum(r, 1e-4)
            cos_t = b.normal * dz / r
            front = cos_t > 0
            sin_t = np.sqrt(np.clip(1 - cos_t * cos_t, 0, 1))
            D = piston_directivity(sin_t) * front
            term = (p0 * b.amp) * D * np.exp(1j * (K * r + ph[None, :])) / r
            p[s:s + chunk] += term.sum(axis=1)
            if want_grad:
                common = term * (1j * K - 1.0 / r) / r
                g[s:s + chunk, 0] += (common * dx).sum(axis=1)
                g[s:s + chunk, 1] += (common * dy).sum(axis=1)
                g[s:s + chunk, 2] += (common * dz).sum(axis=1)
    return p, g


# ----------------------------------------------------------------------------
# Gor'kov potential (+ gravity)
# U = V [ f1 |p|^2 / (4 rho0 c0^2) - f2 * 3 |grad p|^2 / (8 omega^2 rho0) ] + m g z
# ----------------------------------------------------------------------------
class Bead:
    """Small sphere; Gor'kov contrast factors and constants."""
    def __init__(self, diameter_m=2.0e-3, rho=BEAD_RHO, c=BEAD_C):
        self.r = diameter_m / 2
        self.V = 4 / 3 * math.pi * self.r ** 3
        self.m = rho * self.V
        f1 = 1 - (RHO_AIR * C_AIR ** 2) / (rho * c ** 2)
        f2 = 2 * (rho - RHO_AIR) / (2 * rho + RHO_AIR)
        self.k1 = self.V * f1 / (4 * RHO_AIR * C_AIR ** 2)
        self.k2 = self.V * f2 * 3 / (8 * OMEGA ** 2 * RHO_AIR)


def gorkov(boards, pts, p0, bead, gravity=True):
    p, g = field(boards, pts, p0, want_grad=True)
    U = bead.k1 * np.abs(p) ** 2 - bead.k2 * (np.abs(g) ** 2).sum(axis=1)
    if gravity:
        U = U + bead.m * G * np.atleast_2d(pts)[:, 2]
    return U


def force(boards, pt, p0, bead, gravity=True, h=5e-5):
    """F = -grad U by central differences (one vectorised call, 6 points)."""
    offs = np.array([[h, 0, 0], [-h, 0, 0], [0, h, 0], [0, -h, 0], [0, 0, h], [0, 0, -h]])
    U = gorkov(boards, pt[None, :] + offs, p0, bead, gravity)
    return -np.array([U[0] - U[1], U[2] - U[3], U[4] - U[5]]) / (2 * h)


# ----------------------------------------------------------------------------
# Phase solver: focus + optional signature, optional 5-bit quantisation
# ----------------------------------------------------------------------------
SIGNATURES = ("focus", "twin trap", "vortex")


def focus_phases(target, board_z, signature="focus", quantize=False):
    d = np.sqrt((EMX - target[0]) ** 2 + (EMY - target[1]) ** 2 + (board_z - target[2]) ** 2)
    ph = -K * d
    if signature == "twin trap":
        ph = ph + np.pi * (EMX > target[0])
    elif signature == "vortex":
        ph = ph + np.arctan2(EMY - target[1], EMX - target[0])
    ph = np.mod(ph, 2 * np.pi)
    if quantize:  # what the FPGA actually emits: 32 steps per cycle
        ph = np.round(ph / (2 * np.pi) * 32) % 32 * (2 * np.pi / 32)
    return ph


def phases_to_steps(ph):
    return (np.round(np.mod(ph, 2 * np.pi) / (2 * np.pi) * 32) % 32).astype(int)


# ----------------------------------------------------------------------------
# Simulation state
# ----------------------------------------------------------------------------
SETUPS = ("single + reflector", "single, free field", "two opposed boards")


class Sim:
    def __init__(self):
        self.setup = SETUPS[0]
        self.H = 0.160              # board height / separation, m
        self.vpp = 12.0
        self.signature = "vortex"
        self.bd = Bead(2.0e-3)          # bead diameter 2 mm (EPS)
        self.quantize = True
        self.gravity = True
        self.target = np.array([0.0, 0.0, 0.080])
        self.bead = self.target.copy()
        self.dropped = False
        self.dead = set()           # raster indices of disabled emitters
        self.rebuild()

    @property
    def p0(self):
        # square-wave drive: fundamental amplitude is 4/pi of the square wave
        return P0_PER_VOLT * self.vpp * 4 / math.pi

    def enabled_mask(self):
        m = np.ones(N_EM, bool)
        for i in self.dead:
            m[i] = False
        return m

    def rebuild(self):
        t, en = self.target, self.enabled_mask()
        if self.setup == "single + reflector":
            ph = focus_phases(t, self.H, self.signature, self.quantize)
            self.boards = [Board(self.H, -1, ph, enabled=en),
                           Board(-self.H, +1, ph, enabled=en)]      # mirror image
            self.real_boards = [("top", self.H)]
        elif self.setup == "single, free field":
            ph = focus_phases(t, 0.0, self.signature, self.quantize)
            self.boards = [Board(0.0, +1, ph, enabled=en)]
            self.real_boards = [("bottom", 0.0)]
        else:
            pb = focus_phases(t, 0.0, self.signature, self.quantize)
            pt = focus_phases(t, self.H, self.signature, self.quantize)
            self.boards = [Board(0.0, +1, pb, enabled=en), Board(self.H, -1, pt, enabled=en)]
            self.real_boards = [("bottom", 0.0), ("top", self.H)]

    def z_limits(self):
        if self.setup == "single, free field":
            return 0.0, self.H
        return 0.0, self.H

    def step(self, iters=30, max_step=1e-4):
        """
        Quasi-static bead motion: slide downhill on U (Gor'kov + gravity),
        at most max_step metres per iteration. If there is no trap, the bead
        keeps sliding and is flagged as dropped once it leaves the trap region.
        """
        if self.dropped:
            return
        s = getattr(self, "_s", 1e-5)
        U = gorkov(self.boards, self.bead[None, :], self.p0, self.bd, self.gravity)[0]
        for _ in range(iters):
            F = force(self.boards, self.bead, self.p0, self.bd, self.gravity)
            n = np.linalg.norm(F)
            if n == 0:
                break
            trial = self.bead + F / n * s
            Ut = gorkov(self.boards, trial[None, :], self.p0, self.bd, self.gravity)[0]
            if Ut < U:                       # downhill: accept, lengthen step
                self.bead, U = trial, Ut
                s = min(s * 1.5, max_step)
            else:                            # overshot the minimum: shorten step
                s = max(s * 0.5, 1e-8)
        self._s = s
        z0, z1 = self.z_limits()
        if (self.bead[2] < z0 + self.bd.r or self.bead[2] > z1 - self.bd.r
                or np.any(np.abs(self.bead[:2]) > 0.12)
                or np.linalg.norm(self.bead - self.target) > 0.020):
            self.dropped = True

    def reset_bead(self):
        """Place a fresh bead at the target and let it settle into the trap."""
        self.bead = self.target.copy()
        self.dropped = False
        self._s = 1e-5
        for _ in range(40):
            self.step(40, 5e-5)
            if self.dropped:
                break

    def slice_xz(self, n=(150, 120), what="pressure"):
        x = np.linspace(-0.10, 0.10, n[0])
        z0, z1 = self.z_limits()
        z = np.linspace(z0 + 1e-3, z1 - 1e-3, n[1])
        X, Z = np.meshgrid(x, z)
        pts = np.column_stack([X.ravel(), np.full(X.size, self.target[1]), Z.ravel()])
        return x, z, self._eval(pts, what).reshape(Z.shape)

    def slice_xy(self, n=(130, 130), what="pressure"):
        x = np.linspace(-0.10, 0.10, n[0])
        y = np.linspace(-0.10, 0.10, n[1])
        X, Y = np.meshgrid(x, y)
        pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, self.target[2])])
        return x, y, self._eval(pts, what).reshape(X.shape)

    def _eval(self, pts, what):
        if what == "trap potential":
            return gorkov(self.boards, pts, self.p0, self.bd, gravity=False)
        p, _ = field(self.boards, pts, self.p0, want_grad=False)
        return p  # complex; caller picks |p| or Re(p e^{i wt})


# ----------------------------------------------------------------------------
# Plotting (shared by GUI and self-test)
# ----------------------------------------------------------------------------
VIEWS = ("pressure |p|", "waves (animated)", "trap potential")
import matplotlib.patheffects as _pe
_OUTLINE = [_pe.withStroke(linewidth=4, foreground="k")]


class Plotter:
    def __init__(self, fig, sim):
        self.fig, self.sim = fig, sim
        self.ax_side = fig.add_subplot(1, 2, 1)
        self.ax_top = fig.add_subplot(1, 2, 2)
        self.cache = None
        self.im_side = self.im_top = None
        self.cb = None

    def compute(self, view, coarse=False):
        what = "trap potential" if view == "trap potential" else "pressure"
        ns = (70, 56) if coarse else (140, 112)
        nt = (60, 60) if coarse else (120, 120)
        xs, zs, side = self.sim.slice_xz(ns, what)
        xt, yt, top = self.sim.slice_xy(nt, what)
        self.cache = (view, xs, zs, side, xt, yt, top)

    def _img(self, data, view, phase):
        if view == "pressure |p|":
            return np.abs(data) / 1000.0                          # kPa
        if view == "waves (animated)":
            return np.real(data * np.exp(1j * phase)) / 1000.0    # kPa
        u = data * 1e9                                            # nJ
        lo, hi = np.percentile(u, [1, 99])
        return np.clip(u, lo, hi)

    def draw(self, phase=0.0, full=True):
        view, xs, zs, side, xt, yt, top = self.cache
        s_img, t_img = self._img(side, view, phase), self._img(top, view, phase)
        cmap = {"pressure |p|": "inferno", "waves (animated)": "RdBu_r",
                "trap potential": "viridis"}[view]
        if view == "waves (animated)":
            vmax = np.abs(side).max() / 1000.0
            vmin = -vmax
        else:
            vmin, vmax = min(s_img.min(), t_img.min()), max(s_img.max(), t_img.max())
        label = {"pressure |p|": "|p|  (kPa)", "waves (animated)": "p(t)  (kPa)",
                 "trap potential": "U  (nJ)"}[view]
        if not full and self.im_side is not None:
            self.im_side.set_data(s_img)
            self.im_top.set_data(t_img)
            self.im_side.set_clim(vmin, vmax)
            self.im_top.set_clim(vmin, vmax)
            self._overlays()
            return
        mm = 1e3
        for ax in (self.ax_side, self.ax_top):
            ax.clear()
        self.im_side = self.ax_side.imshow(
            s_img, origin="lower", aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax,
            extent=[xs[0] * mm, xs[-1] * mm, zs[0] * mm, zs[-1] * mm], interpolation="bilinear")
        self.im_top = self.ax_top.imshow(
            t_img, origin="lower", aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax,
            extent=[xt[0] * mm, xt[-1] * mm, yt[0] * mm, yt[-1] * mm], interpolation="bilinear")
        self.ax_side.set_xlabel("x (mm)")
        self.ax_side.set_ylabel("z (mm)")
        self.ax_top.set_xlabel("x (mm)")
        self.ax_top.set_ylabel("y (mm)")
        if self.cb is None:
            self.cb = self.fig.colorbar(self.im_top, ax=[self.ax_side, self.ax_top],
                                        shrink=0.8, pad=0.02)
        else:
            self.cb.update_normal(self.im_top)
        self.cb.set_label(label)
        self._static_overlays()
        self._overlays()

    def save_gif(self, fn, frames=24, fps=12, dpi=80):
        """One acoustic cycle of the 'waves' view as a looping GIF (needs Pillow)."""
        from matplotlib.animation import PillowWriter
        prev = self.cache[0] if self.cache else None
        if prev != "waves (animated)":
            self.compute("waves (animated)")
        self.draw(0.0, full=True)
        writer = PillowWriter(fps=fps)
        with writer.saving(self.fig, fn, dpi):
            for k in range(frames):
                self.draw(2 * math.pi * k / frames, full=False)
                writer.grab_frame()
        if prev is not None and prev != "waves (animated)":
            self.compute(prev)
        self.draw(0.0, full=True)

    def _static_overlays(self):
        s, mm = self.sim, 1e3
        half = (N_SIDE - 1) / 2 * PITCH * mm + 5
        for name, z in s.real_boards:
            self.ax_side.plot([-half, half], [z * mm, z * mm], color="#58c4dd", lw=4,
                              solid_capstyle="butt")
        if s.setup == "single + reflector":
            self.ax_side.plot([-100, 100], [0.5, 0.5], color="0.75", lw=3)
        self.ax_top.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
            (-half, -half), 2 * half, 2 * half, fill=False, ec="#58c4dd", lw=1, ls="--"))
        self.side_marks = [
            self.ax_side.plot([], [], "x", color="w", ms=9, mew=2, path_effects=_OUTLINE)[0],
            self.ax_side.plot([], [], "o", mfc="w", mec="k", ms=7)[0]]
        self.top_marks = [
            self.ax_top.plot([], [], "x", color="w", ms=9, mew=2, path_effects=_OUTLINE)[0],
            self.ax_top.plot([], [], "o", mfc="w", mec="k", ms=7)[0]]
        s_ = self.sim
        self.ax_side.set_title(f"Side view (x-z) at y = {s_.target[1]*mm:.1f} mm", fontsize=10)
        self.ax_top.set_title(f"Top view (x-y) at z = {s_.target[2]*mm:.1f} mm", fontsize=10)
        self.ax_top.legend([self.top_marks[0], self.top_marks[1]], ["target (commanded)", "bead"],
                           loc="lower right", fontsize=8, framealpha=0.85)

    def _overlays(self):
        s, mm = self.sim, 1e3
        t, b = s.target * mm, s.bead * mm
        self.side_marks[0].set_data([t[0]], [t[2]])
        self.top_marks[0].set_data([t[0]], [t[1]])
        if s.dropped:
            self.side_marks[1].set_data([], [])
            self.top_marks[1].set_data([], [])
        else:
            self.side_marks[1].set_data([b[0]], [b[2]])
            self.top_marks[1].set_data([b[0]], [b[1]])


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------
def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    sim = Sim()
    root = tk.Tk()
    root.title("HAL levitation simulator")

    fig = Figure(figsize=(11, 5.2), dpi=100, layout="constrained")
    plot = Plotter(fig, sim)
    canvas = FigureCanvasTkAgg(fig, master=root)

    # ---- controls ----
    ctl = ttk.Frame(root, padding=6)
    ctl.pack(side=tk.LEFT, fill=tk.Y)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    NavigationToolbar2Tk(canvas, root).update()

    def row(label, widget):
        f = ttk.Frame(ctl)
        f.pack(fill=tk.X, pady=2)
        ttk.Label(f, text=label, width=16).pack(side=tk.LEFT)
        widget(f).pack(side=tk.LEFT, fill=tk.X, expand=True)

    v_setup = tk.StringVar(value=sim.setup)
    v_sig = tk.StringVar(value=sim.signature)
    v_view = tk.StringVar(value=VIEWS[0])
    v_H = tk.StringVar(value=f"{sim.H*1e3:.0f}")
    v_vpp = tk.StringVar(value=f"{sim.vpp:.1f}")
    v_bead = tk.StringVar(value=f"{sim.bd.r*2e3:.1f}")
    v_q = tk.BooleanVar(value=sim.quantize)
    v_g = tk.BooleanVar(value=sim.gravity)
    v_x = tk.StringVar(value="0")
    v_y = tk.StringVar(value="0")
    v_z = tk.StringVar(value=f"{sim.target[2]*1e3:.0f}")
    v_speed = tk.StringVar(value="20")
    v_dead = tk.StringVar(value="")
    status = tk.StringVar()

    ttk.Label(ctl, text="Setup", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
    row("Arrangement", lambda f: ttk.Combobox(f, textvariable=v_setup, values=SETUPS,
                                              state="readonly", width=20))
    row("Board height/gap (mm)", lambda f: ttk.Entry(f, textvariable=v_H, width=8))
    row("Drive (Vpp)", lambda f: ttk.Entry(f, textvariable=v_vpp, width=8))
    row("Bead dia. (mm)", lambda f: ttk.Entry(f, textvariable=v_bead, width=8))
    row("Signature", lambda f: ttk.Combobox(f, textvariable=v_sig, values=SIGNATURES,
                                            state="readonly", width=20))
    ttk.Checkbutton(ctl, text="5-bit phases (as FPGA)", variable=v_q).pack(anchor="w")
    ttk.Checkbutton(ctl, text="Gravity", variable=v_g).pack(anchor="w")
    row("Dead emitters", lambda f: ttk.Entry(f, textvariable=v_dead, width=20))
    ttk.Label(ctl, text="  raster index 0-255, comma separated", foreground="gray").pack(anchor="w")

    ttk.Separator(ctl).pack(fill=tk.X, pady=6)
    ttk.Label(ctl, text="Move trap to (mm)", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
    row("x", lambda f: ttk.Entry(f, textvariable=v_x, width=8))
    row("y", lambda f: ttk.Entry(f, textvariable=v_y, width=8))
    row("z", lambda f: ttk.Entry(f, textvariable=v_z, width=8))
    row("Speed (mm/s)", lambda f: ttk.Entry(f, textvariable=v_speed, width=8))

    ttk.Separator(ctl).pack(fill=tk.X, pady=6)
    row("View", lambda f: ttk.Combobox(f, textvariable=v_view, values=VIEWS,
                                       state="readonly", width=20))

    state = {"move": None, "wt": 0.0, "dirty": False, "last_field": 0.0}
    RECOMMENDED = {"single + reflector": "vortex", "single, free field": "twin trap",
                   "two opposed boards": "focus"}
    v_setup.trace_add("write", lambda *_: v_sig.set(RECOMMENDED[v_setup.get()]))

    def apply_setup(*_):
        try:
            sim.H = float(v_H.get()) / 1e3
            sim.vpp = float(v_vpp.get())
            sim.bd = Bead(float(v_bead.get()) / 1e3)
            sim.dead = {int(s) for s in v_dead.get().replace(" ", "").split(",") if s != ""}
        except ValueError:
            messagebox.showerror("Input", "Board height, drive, bead size and dead emitters must be numbers.")
            return
        sim.setup = v_setup.get()
        sim.signature = v_sig.get()
        sim.quantize = v_q.get()
        sim.gravity = v_g.get()
        sim.rebuild()
        sim.reset_bead()
        refresh(full=True)

    def move(*_):
        try:
            tgt = np.array([float(v_x.get()), float(v_y.get()), float(v_z.get())]) / 1e3
            speed = max(float(v_speed.get()), 1.0) / 1e3
        except ValueError:
            messagebox.showerror("Input", "x, y, z and speed must be numbers (mm).")
            return
        start = sim.target.copy()
        dur = np.linalg.norm(tgt - start) / speed
        state["move"] = (start, tgt, time.time(), max(dur, 1e-3))

    def drop_new(*_):
        sim.reset_bead()
        refresh(full=True)

    def save_svg():
        fn = filedialog.asksaveasfilename(defaultextension=".svg",
                                          filetypes=[("SVG", "*.svg"), ("PNG", "*.png"), ("PDF", "*.pdf")])
        if fn:
            fig.savefig(fn)

    def save_gif():
        fn = filedialog.asksaveasfilename(defaultextension=".gif", filetypes=[("GIF", "*.gif")])
        if not fn:
            return
        status.set("Rendering GIF...")
        root.update_idletasks()
        plot.save_gif(fn)
        canvas.draw_idle()

    def save_phases():
        fn = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not fn:
            return
        with open(fn, "w") as f:
            f.write("board,raster_index,row,col,x_mm,y_mm,phase_rad,phase_step_0_31\n")
            names = [n for n, _ in sim.real_boards]
            for bi, (b, name) in enumerate(zip(sim.boards, names)):
                steps = phases_to_steps(b.phases)
                for i in range(N_EM):
                    f.write(f"{name},{i},{i // N_SIDE},{i % N_SIDE},{EMX[i]*1e3:.2f},{EMY[i]*1e3:.2f},"
                            f"{b.phases[i]:.5f},{steps[i]}\n")

    ttk.Button(ctl, text="Apply setup", command=apply_setup).pack(fill=tk.X, pady=(8, 2))
    ttk.Button(ctl, text="Move trap", command=move).pack(fill=tk.X, pady=2)
    ttk.Button(ctl, text="Drop new bead", command=drop_new).pack(fill=tk.X, pady=2)
    ttk.Button(ctl, text="Save figure (SVG)", command=save_svg).pack(fill=tk.X, pady=2)
    ttk.Button(ctl, text="Export phases (CSV)", command=save_phases).pack(fill=tk.X, pady=2)
    ttk.Button(ctl, text="Export wave GIF", command=save_gif).pack(fill=tk.X, pady=2)
    ttk.Label(ctl, textvariable=status, wraplength=230, justify=tk.LEFT,
              font=("TkFixedFont", 9)).pack(anchor="w", pady=8)
    v_view.trace_add("write", lambda *_: refresh(full=True))

    def refresh(full=True, coarse=False):
        plot.compute(v_view.get(), coarse=coarse)
        plot.draw(state["wt"], full=full)
        canvas.draw_idle()

    def update_status():
        t, b = sim.target * 1e3, sim.bead * 1e3
        err = np.linalg.norm(sim.bead - sim.target) * 1e3
        s = (f"target  ({t[0]:6.1f},{t[1]:6.1f},{t[2]:6.1f}) mm\n"
             f"bead    ({b[0]:6.1f},{b[1]:6.1f},{b[2]:6.1f}) mm\n"
             f"offset  {err:5.2f} mm\n"
             f"lambda/2 = {LAMBDA/2*1e3:.2f} mm")
        if sim.dropped:
            s += "\n\nBEAD DROPPED - no trap here.\nTry another point or 'Drop new bead'."
        status.set(s)

    FRAME = 0.04          # s between GUI frames
    FIELD_EVERY = 0.35    # s between (coarse) heat-map updates while moving

    def tick():
        now = time.time()
        mv = state["move"]
        if mv is not None:
            start, tgt, t0, dur = mv
            a = min((now - t0) / dur, 1.0)
            sim.target = start + (tgt - start) * a      # FPGA would re-solve phases continuously
            sim.rebuild()
            if a >= 1.0:
                state["move"] = None
                state["dirty"] = True                   # final full-res redraw
            elif now - state["last_field"] > FIELD_EVERY:
                plot.compute(v_view.get(), coarse=True)
                plot.draw(state["wt"], full=False)
                state["last_field"] = time.time()
        sim.step(30, 1e-4)
        if state["dirty"]:
            state["dirty"] = False
            refresh(full=True)
        elif v_view.get() == "waves (animated)":
            state["wt"] = (state["wt"] + 2 * math.pi / 24) % (2 * math.pi)
            plot.draw(state["wt"], full=False)
        else:
            plot._overlays()
        update_status()
        canvas.draw_idle()
        root.after(int(FRAME * 1000), tick)

    sim.reset_bead()
    refresh(full=True)
    root.after(100, tick)
    root.mainloop()


# ----------------------------------------------------------------------------
# Batch rendering: one image per (arrangement, point), no GUI
# ----------------------------------------------------------------------------
SETUP_FOLDERS = {"single + reflector": "single_board_reflector",
                 "single, free field": "single_board_free",
                 "two opposed boards": "two_opposed_boards"}
RECOMMENDED_SIGNATURE = {"single + reflector": "vortex", "single, free field": "twin trap",
                         "two opposed boards": "focus"}
VIEW_FOLDERS = {"pressure |p|": "pressure", "waves (animated)": "waves",
                "trap potential": "trap_potential"}


def raster_points(xs, ys, zs):
    """All (x, y, z) combinations in mm, raster order: z top-down, then y, then x."""
    return [(x, y, z) for z in zs for y in ys for x in xs]


def diagonal_points(start, stop, step, zs):
    """(v, v, z) along the x = y diagonal for each z, in mm."""
    vals = np.arange(start, stop + step / 2, step)
    return [(v, v, z) for z in zs for v in vals]


def _fmt(v):
    return f"{v:g}".replace(".", "p")


def point_name(p):
    """Filename stem for a point in mm, e.g. x-80_y-40_z120."""
    return f"x{_fmt(p[0])}_y{_fmt(p[1])}_z{_fmt(p[2])}"


def batch_render(points_mm, outdir="sim_output", setups=SETUPS, views=("pressure |p|",),
                 signatures=None, vpp=12.0, bead_mm=2.0, gap_mm=160.0, quantize=True,
                 gravity=True, dpi=150, fmt="png", gif_frames=24, verbose=True):
    """
    Render every point in every arrangement and save the images.

    points_mm  : list of (x, y, z) in mm
    outdir     : root folder; one subfolder per arrangement (and per view if
                 more than one view is requested)
    setups     : any subset of SETUPS
    views      : any subset of VIEWS ("waves (animated)" saves a still unless
                 fmt="gif", which saves the one-cycle animation)
    signatures : dict setup -> trap shape; defaults to the recommended ones
    Also writes summary.csv per arrangement: target, settled bead, offset, held.
    Returns a list of result dicts.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    sigs = dict(RECOMMENDED_SIGNATURE)
    if signatures:
        sigs.update(signatures)
    results = []
    total = len(points_mm) * len(setups)
    n = 0
    t_start = time.time()
    for setup in setups:
        folder = os.path.join(outdir, SETUP_FOLDERS[setup])
        os.makedirs(folder, exist_ok=True)
        sim = Sim()
        sim.setup, sim.signature = setup, sigs[setup]
        sim.vpp, sim.bd, sim.H = vpp, Bead(bead_mm / 1e3), gap_mm / 1e3
        sim.quantize, sim.gravity = quantize, gravity
        fig = Figure(figsize=(11, 5.2), layout="constrained")
        plot = Plotter(fig, sim)
        rows = []
        for p in points_mm:
            n += 1
            sim.target = np.array(p, float) / 1e3
            sim.rebuild()
            sim.reset_bead()
            held = not sim.dropped
            off = (sim.bead - sim.target) * 1e3
            for view in views:
                vdir = folder if len(views) == 1 else os.path.join(folder, VIEW_FOLDERS[view])
                os.makedirs(vdir, exist_ok=True)
                fn = os.path.join(vdir, point_name(p) + "." + fmt)
                if fmt == "gif":
                    plot.cache = None
                    plot.compute(view)
                    if view == "waves (animated)":
                        plot.save_gif(fn, frames=gif_frames)
                        continue
                    plot.draw(0.0, full=True)
                    fig.savefig(fn.replace(".gif", ".png"), dpi=dpi)
                    continue
                plot.compute(view)
                plot.draw(0.0, full=True)
                fig.suptitle(f"{setup}, {sigs[setup]} | target ({p[0]:g}, {p[1]:g}, {p[2]:g}) mm | "
                             + ("bead held, offset (%.1f, %.1f, %.1f) mm" % tuple(off) if held
                                else "bead NOT held"), fontsize=10)
                fig.savefig(fn, dpi=dpi)
                fig.suptitle("")
            rows.append((p, sim.bead * 1e3, off, held))
            results.append({"setup": setup, "target_mm": p, "bead_mm": tuple(sim.bead * 1e3),
                            "held": held})
            if verbose:
                el = time.time() - t_start
                print(f"[{n}/{total}] {SETUP_FOLDERS[setup]:22s} {point_name(p):22s} "
                      f"{'held' if held else 'DROPPED':7s}  ~{el / n * (total - n) / 60:.1f} min left",
                      flush=True)
        with open(os.path.join(folder, "summary.csv"), "w") as f:
            f.write("x_mm,y_mm,z_mm,bead_x_mm,bead_y_mm,bead_z_mm,offset_x_mm,offset_y_mm,"
                    "offset_z_mm,held\n")
            for p, b, o, h in rows:
                f.write(",".join(f"{v:.3f}" for v in (*p, *b, *o)) + f",{int(h)}\n")
    return results


# ----------------------------------------------------------------------------
# Headless self-test: settle a bead in each setup, save a figure
# ----------------------------------------------------------------------------
def selftest():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    for setup in SETUPS:
        for sig in SIGNATURES:
            s = Sim()
            z = 0.060 if "free" in setup else 0.080
            s.setup, s.signature, s.target = setup, sig, np.array([0.010, 0.005, z])
            t0 = time.time()
            s.rebuild()
            s.reset_bead()
            off = (s.bead - s.target) * 1e3
            print(f"{setup:20s} {sig:9s} offset from target {np.round(off, 2)} mm  "
                  f"held={not s.dropped}  ({time.time()-t0:.2f}s)")
    s = Sim()
    s.rebuild(); s.reset_bead()
    fig = Figure(figsize=(11, 5.2), layout="constrained")
    pl = Plotter(fig, s)
    for view, fn in [("pressure |p|", "selftest_pressure.svg"),
                     ("trap potential", "selftest_trap.png"),
                     ("waves (animated)", "selftest_waves.png")]:
        t0 = time.time()
        pl.compute(view); pl.draw()
        print(f"{view:18s} full-res compute+draw {time.time()-t0:.2f}s")
        fig.savefig(fn, dpi=80)
    t0 = time.time(); pl.save_gif("selftest_waves.gif")
    print(f"GIF export {time.time()-t0:.2f}s")
    t0 = time.time(); pl.compute("pressure |p|", coarse=True)
    print(f"coarse field update {time.time()-t0:.2f}s")
    t0 = time.time(); s.step(30, 1e-4)
    print(f"one GUI frame of bead motion: {(time.time()-t0)*1e3:.1f} ms")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run_gui()