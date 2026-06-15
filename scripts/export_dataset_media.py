from pathlib import Path
from io import BytesIO
import argparse
import json
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
VIEWS = {
    'observation.image': 'agent',
    'observation.wrist_image': 'wrist',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Export LeRobot parquet images to MP4 and GIF.')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'demo_data')
    parser.add_argument('--out-root', type=Path, default=ROOT / 'media' / 'trajectories')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--gif-fps', type=int, default=12)
    parser.add_argument('--gif-width', type=int, default=480)
    parser.add_argument('--ffmpeg', type=Path, default=None)
    return parser.parse_args()


def resolve_ffmpeg(ffmpeg_path):
    if ffmpeg_path is not None:
        return str(ffmpeg_path)
    detected = shutil.which('ffmpeg')
    if detected:
        return detected
    fallback = Path('/home/p/miniconda3/bin/ffmpeg')
    if fallback.exists():
        return str(fallback)
    raise SystemExit('ffmpeg not found. Install ffmpeg or pass --ffmpeg /path/to/ffmpeg')


def load_dataset(data_root):
    import pandas as pd

    frames = []
    for parquet_path in sorted((data_root / 'data').glob('chunk-*/*.parquet')):
        df = pd.read_parquet(parquet_path)
        df['_source'] = str(parquet_path.relative_to(ROOT))
        frames.append(df)
    if not frames:
        raise SystemExit(f'No parquet files found under {data_root / "data"}')
    return pd.concat(frames, ignore_index=True)


def main():
    args = parse_args()
    from PIL import Image

    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    args.out_root.mkdir(parents=True, exist_ok=True)
    all_df = load_dataset(args.data_root)

    summary = []
    for episode_index in sorted(all_df['episode_index'].unique()):
        ep_df = all_df[all_df['episode_index'] == episode_index].sort_values('frame_index')
        human_episode = int(episode_index) + 1
        for column, view_name in VIEWS.items():
            out_dir = args.out_root / f'episode_{human_episode:02d}'
            out_dir.mkdir(parents=True, exist_ok=True)
            mp4_path = out_dir / f'episode_{human_episode:02d}_{view_name}.mp4'
            gif_path = out_dir / f'episode_{human_episode:02d}_{view_name}.gif'
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                for frame_number, cell in enumerate(ep_df[column].tolist()):
                    img = Image.open(BytesIO(cell['bytes'])).convert('RGB')
                    img.save(tmp_path / f'frame_{frame_number:06d}.png')
                subprocess.run([
                    ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
                    '-framerate', str(args.fps),
                    '-i', str(tmp_path / 'frame_%06d.png'),
                    '-vf', 'format=yuv420p',
                    '-movflags', '+faststart',
                    str(mp4_path),
                ], check=True)
                palette_path = tmp_path / 'palette.png'
                subprocess.run([
                    ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', str(mp4_path),
                    '-vf', f'fps={args.gif_fps},scale={args.gif_width}:-1:flags=lanczos,palettegen',
                    str(palette_path),
                ], check=True)
                subprocess.run([
                    ffmpeg, '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', str(mp4_path),
                    '-i', str(palette_path),
                    '-lavfi', f'fps={args.gif_fps},scale={args.gif_width}:-1:flags=lanczos[x];[x][1:v]paletteuse',
                    str(gif_path),
                ], check=True)
            summary.append({
                'episode': human_episode,
                'episode_index': int(episode_index),
                'view': view_name,
                'frames': int(len(ep_df)),
                'duration_sec': round(float(len(ep_df)) / args.fps, 3),
                'mp4': str(mp4_path.relative_to(ROOT)),
                'gif': str(gif_path.relative_to(ROOT)),
                'mp4_size_mb': round(mp4_path.stat().st_size / 1024 / 1024, 3),
                'gif_size_mb': round(gif_path.stat().st_size / 1024 / 1024, 3),
            })

    summary_path = args.out_root / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
