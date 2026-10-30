#!/usr/bin/env python3
"""
Turn a GIF into ONE multi-page PDF for LaTeX's animate package.

Instead of uploading hundreds of PNG frames to Overleaf, upload one PDF.
Optionally keep only every Nth frame and/or only the last part of the clip.

Usage:
    python gif2pdf.py spiral.gif                     # every frame -> spiral_frames.pdf
    python gif2pdf.py spiral.gif --max-frames 60     # thin it out to <= 60 frames
    python gif2pdf.py spiral.gif --last 1.5          # only the last 1.5 s (e.g. the fall)
    python gif2pdf.py spiral.gif --last 1.5 --max-frames 60

It prints the \\animategraphics line to paste, with the right frame rate.

In LaTeX (preamble, after graphicx):  \\usepackage{animate}
"""

import argparse
import math
import os

from PIL import Image, ImageSequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gif")
    ap.add_argument("--max-frames", type=int, default=None, help="keep at most this many frames")
    ap.add_argument("--last", type=float, default=None, help="keep only the last N seconds of the GIF")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    im = Image.open(a.gif)
    frames, durs = [], []
    for fr in ImageSequence.Iterator(im):
        frames.append(fr.convert("RGB"))
        durs.append(fr.info.get("duration", im.info.get("duration", 40)) or 40)
    dt = sum(durs) / len(durs) / 1000.0            # s per frame in the GIF
    fps_in = 1.0 / dt

    if a.last is not None:
        n_keep = max(1, int(round(a.last / dt)))
        frames = frames[-n_keep:]
    step = 1
    if a.max_frames and len(frames) > a.max_frames:
        step = math.ceil(len(frames) / a.max_frames)
        frames = frames[::step]
    fps_out = fps_in / step

    out = a.out or os.path.splitext(a.gif)[0] + "_frames.pdf"
    frames[0].save(out, save_all=True, append_images=frames[1:], resolution=150)
    name = os.path.splitext(os.path.basename(out))[0]
    print(f"{len(frames)} frames -> {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    print("\nPaste into your figure:")
    print(f"\\animategraphics[width=\\columnwidth,loop,autoplay,controls]{{{fps_out:.4g}}}{{{name}}}{{}}{{}}")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Turn a GIF into ONE multi-page PDF for LaTeX's animate package.

Instead of uploading hundreds of PNG frames to Overleaf, upload one PDF.
Optionally keep only every Nth frame and/or only the last part of the clip.

Usage:
    python gif2pdf.py spiral.gif                     # every frame -> spiral_frames.pdf
    python gif2pdf.py spiral.gif --max-frames 60     # thin it out to <= 60 frames
    python gif2pdf.py spiral.gif --last 1.5          # only the last 1.5 s (e.g. the fall)
    python gif2pdf.py spiral.gif --last 1.5 --max-frames 60

It prints the \\animategraphics line to paste, with the right frame rate.

In LaTeX (preamble, after graphicx):  \\usepackage{animate}
"""

import argparse
import math
import os

from PIL import Image, ImageSequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gif")
    ap.add_argument("--max-frames", type=int, default=None, help="keep at most this many frames")
    ap.add_argument("--last", type=float, default=None, help="keep only the last N seconds of the GIF")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    im = Image.open(a.gif)
    frames, durs = [], []
    for fr in ImageSequence.Iterator(im):
        frames.append(fr.convert("RGB"))
        durs.append(fr.info.get("duration", im.info.get("duration", 40)) or 40)
    dt = sum(durs) / len(durs) / 1000.0            # s per frame in the GIF
    fps_in = 1.0 / dt

    if a.last is not None:
        n_keep = max(1, int(round(a.last / dt)))
        frames = frames[-n_keep:]
    step = 1
    if a.max_frames and len(frames) > a.max_frames:
        step = math.ceil(len(frames) / a.max_frames)
        frames = frames[::step]
    fps_out = fps_in / step

    out = a.out or os.path.splitext(a.gif)[0] + "_frames.pdf"
    frames[0].save(out, save_all=True, append_images=frames[1:], resolution=150)
    name = os.path.splitext(os.path.basename(out))[0]
    print(f"{len(frames)} frames -> {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    print("\nPaste into your figure:")
    print(f"\\animategraphics[width=\\columnwidth,loop,autoplay,controls]{{{fps_out:.4g}}}{{{name}}}{{}}{{}}")


if __name__ == "__main__":
    main()