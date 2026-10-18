"""Make a synchronized six-panel preview from the saved GT gallery videos."""
from contextlib import ExitStack
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    gallery = Path("docs/visualizations/gt-toss")
    output = Path("docs/assets")
    output.mkdir(exist_ok=True)
    font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    font = ImageFont.truetype(str(font_path), 18)
    small_font = ImageFont.truetype(str(font_path), 14)
    frames = []
    with ExitStack() as stack:
        readers = [stack.enter_context(imageio.get_reader(gallery / f"trajectory_{i:05d}/gt.mp4")) for i in range(6)]
        source_rate = readers[0].get_meta_data()["fps"]
        count = min(reader.count_frames() for reader in readers)
        for index in range(0, count, 2):
            canvas = Image.new("RGB", (960, 560), "#112b3a")
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 9), "RANDOM CUBE TOSS  /  ALL ACTIONS = 0", font=font, fill="#80e3e8")
            draw.text((16, 39), "Six recorded trajectories; same physical time; each panel has a GT-fitted camera", font=small_font, fill="white")
            for identity, reader in enumerate(readers):
                frame = Image.fromarray(reader.get_data(index)).resize((320, 240), Image.Resampling.LANCZOS)
                x, y = (identity % 3)*320, 80 + (identity//3)*240
                canvas.paste(frame, (x, y))
                draw.rectangle((x+215, y, x+320, y+24), fill="#112b3a")
                draw.text((x+220, y+3), f"ID {identity:05d}", font=small_font, fill="white")
            frames.append(canvas)
    duration_ms = round(2000/source_rate)
    frames[0].save(output / "cube-toss-preview.gif", save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    frames[0].save(output / "cube-toss-preview.png")
    with imageio.get_writer(output / "cube-toss-preview.mp4", fps=source_rate/2, codec="libx264", quality=7) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame))
    print({"preview": str(output / "cube-toss-preview.gif"), "frames": len(frames), "fps": source_rate/2})


if __name__ == "__main__":
    main()
