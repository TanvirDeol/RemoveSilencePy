# Standalone Jumpcutter

A complete, self-contained Python implementation for removing silent parts from videos. This tool analyzes video audio to detect silent segments and either removes them entirely or speeds them up, creating jumpcut-style videos.

## Features

- **Silent Part Detection**: Automatically detects silent segments based on audio magnitude analysis
- **Flexible Cutting**: Remove silent parts completely or speed them up
- **Configurable Parameters**: Fine-tune silence detection thresholds and processing options
- **Multiple Output Modes**: Cut silent parts, voiced parts, or both
- **Progress Tracking**: Visual progress bars for long processing operations
- **Command Line Interface**: Easy-to-use CLI with comprehensive options

## Installation

1. **Clone or download** the `standalone_jumpcutter.py` file
2. **Install dependencies**:
   ```bash
   pip install -r standalone_requirements.txt
   ```

   Or install manually:
   ```bash
   pip install moviepy numpy tqdm
   ```

## Usage

### Basic Usage

```bash
# Remove silent parts from a video
python standalone_jumpcutter.py input_video.mp4 output_video.mp4
```

### Advanced Usage

```bash
# Custom silence detection thresholds
python standalone_jumpcutter.py input.mp4 output.mp4 \
    --magnitude-threshold 0.01 \
    --duration-threshold 2.0

# Speed up silence instead of removing it
python standalone_jumpcutter.py input.mp4 output.mp4 \
    --silence-speed 3

# Cut both silent and voiced parts (creates two output files)
python standalone_jumpcutter.py input.mp4 output.mp4 \
    --cut both

# Custom buffer around silence intervals
python standalone_jumpcutter.py input.mp4 output.mp4 \
    --space-on-edges 0.5
```

### Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `input` | str | - | Input video file path |
| `output` | str | - | Output video file path |
| `--magnitude-threshold` | float | 0.02 | Audio magnitude threshold for silence detection |
| `--duration-threshold` | float | 1.0 | Minimum silence duration to cut (seconds) |
| `--failure-tolerance` | float | 0.1 | Tolerance for non-silent samples |
| `--space-on-edges` | float | 0.2 | Buffer time around silence intervals (seconds) |
| `--silence-speed` | int | None | Speed multiplier for silent parts (None = remove) |
| `--min-loud-duration` | int | -1 | Minimum duration for loud parts to keep |
| `--codec` | str | libx264 | Video codec for output |
| `--bitrate` | str | None | Video bitrate for output |
| `--cut` | str | silent | Type of cutting: silent, voiced, or both |

## How It Works

1. **Audio Analysis**: The tool extracts audio from the video and analyzes its magnitude over time
2. **Silence Detection**: It identifies segments where audio magnitude falls below the threshold for a specified duration
3. **Interval Processing**: Silent intervals are identified with configurable buffer zones
4. **Video Processing**: The video is processed to either:
   - Remove silent parts completely (default)
   - Speed up silent parts by a specified factor
   - Keep only voiced parts (opposite of silent)
5. **Output Generation**: The processed video is saved with the specified codec and settings

## Parameters Explained

### Magnitude Threshold
- **Lower values** (e.g., 0.01): More sensitive to quiet sounds, cuts more aggressively
- **Higher values** (e.g., 0.05): Less sensitive, only cuts very quiet parts

### Duration Threshold
- **Shorter durations** (e.g., 0.5s): Cuts brief pauses and hesitations
- **Longer durations** (e.g., 2.0s): Only cuts longer silent segments

### Space on Edges
- **Smaller values** (e.g., 0.1s): Tighter cuts around silence
- **Larger values** (e.g., 0.5s): More conservative cuts with buffer zones

### Silence Speed
- **None**: Completely remove silent parts
- **Integer values** (e.g., 3): Speed up silent parts by 3x

## Examples

### Podcast/Interview Processing
```bash
# Remove long pauses and "ums" from a podcast
python standalone_jumpcutter.py podcast.mp4 podcast_clean.mp4 \
    --magnitude-threshold 0.015 \
    --duration-threshold 0.8 \
    --space-on-edges 0.1
```

### Lecture/Educational Content
```bash
# Speed up silent parts instead of removing them
python standalone_jumpcutter.py lecture.mp4 lecture_condensed.mp4 \
    --silence-speed 4 \
    --duration-threshold 1.5
```

### Music Video Processing
```bash
# More conservative cutting for music content
python standalone_jumpcutter.py music_video.mp4 music_clean.mp4 \
    --magnitude-threshold 0.05 \
    --duration-threshold 2.0 \
    --space-on-edges 0.3
```

## Requirements

- Python 3.7+
- FFmpeg (for video processing)
- Required Python packages:
  - moviepy
  - numpy
  - tqdm

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Install FFmpeg and ensure it's in your system PATH
2. **Memory issues with large videos**: Process shorter segments or reduce video quality
3. **Audio codec issues**: Try different audio codecs or ensure input has audio track

### Performance Tips

- Use lower resolution videos for faster processing
- Adjust `--duration-threshold` to reduce processing time
- Consider using `--silence-speed` instead of complete removal for better performance

## License

This project is open source. Feel free to use, modify, and distribute according to your needs.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Changelog

- **v1.0.0**: Initial standalone release with core jumpcutter functionality
