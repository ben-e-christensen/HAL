from PIL import Image, ImageSequence
import os
os.makedirs("frames", exist_ok=True)
for i, fr in enumerate(ImageSequence.Iterator(Image.open("spiral.gif"))):
    fr.convert("RGB").save(f"frames/f_{i:03d}.png")