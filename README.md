# Remove Silence from Videos

A simple Python tool that automatically removes silent parts from videos. Perfect for cleaning up recordings, podcasts, or any video content with unwanted pauses.

## Features

- **Automatic Silence Detection**: Analyzes audio to find silent segments
- **Simple Usage**: Just provide an input video file
- **Smart Defaults**: Uses sensible parameters that work for most videos
- **Progress Tracking**: Shows progress bars during processing
- **Clean Output**: Creates a new video with "_jumpcut" added to the filename

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install moviepy numpy tqdm
   ```

2. **Make sure FFmpeg is installed** on your system

## Usage

### Basic Usage

```bash
# Remove silent parts from a video
python remvove_silence.py input_video.mp4
```

The tool will automatically:
- Analyze the video's audio track
- Find silent segments (pauses longer than 0.5 seconds)
- Remove those segments
- Save the result as `input_video_jumpcut.mp4`

### Example

```bash
python remvove_silence.py my_podcast.mp4
# Creates: my_podcast_jumpcut.mp4
```

## How It Works

1. **Audio Analysis**: Extracts audio from the video and analyzes its volume levels
2. **Silence Detection**: Identifies segments where audio is very quiet for a sustained period
3. **Smart Cutting**: Removes silent parts while keeping small buffer zones to avoid cutting too close to speech
4. **Video Reconstruction**: Combines the remaining audio/video segments into a new file

## Default Settings

The tool uses these sensible defaults:
- **Silence threshold**: 2% of maximum audio level
- **Minimum silence duration**: 0.5 seconds
- **Buffer zones**: 0.2 seconds around each silence interval
- **Minimum clip length**: 0.1 seconds (prevents very short clips)

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
2. **No audio track**: Make sure your video has an audio track
3. **Memory issues**: For very large videos, consider processing shorter segments

### Tips

- Works best with videos that have clear speech and distinct silent periods
- For music videos or content with background music, the tool may be too aggressive
- The output video will be shorter than the original due to removed silence
