"""
Simple Jumpcutter - Remove Silent Parts from Videos

A minimal implementation for removing silent parts from videos.
Only requires input video path - uses sensible defaults for all other parameters.

Usage:
    python jumpcutter.py input_video.mp4
"""

import os
import sys
from pathlib import Path

import numpy as np
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.editor import VideoFileClip, concatenate_videoclips
from tqdm import tqdm


def block_printing(func):
    """
    Decorator to temporarily suppress print statements during function execution.
    Useful for cleaning up output from third-party libraries.
    """
    def func_wrapper(*args, **kwargs):
        # Block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # Call the method in question
        value = func(*args, **kwargs)
        # Enable all printing to the console
        sys.stdout = sys.__stdout__
        # Pass the return value of the method back
        return value
    return func_wrapper


class Audio:
    """
    Audio analysis class for detecting silent segments in video audio.
    """
    
    def __init__(self, audio: AudioFileClip) -> None:
        self.audio = audio
        self.fps = audio.fps
        self.signal = self.audio.to_soundarray()
        if len(self.signal.shape) == 1:
            self.signal = self.signal.reshape(-1, 1)

    def get_intervals_to_cut(self) -> list:
        """Analyze audio to find silent intervals to cut."""
        # Use sensible defaults
        magnitude_threshold_ratio = 0.02
        duration_threshold_in_seconds = 0.5
        failure_tolerance_ratio = 0.1
        space_on_edges = 0.2
        
        min_magnitude = min(abs(np.min(self.signal)), np.max(self.signal))
        magnitude_threshold = min_magnitude * magnitude_threshold_ratio
        failure_tolerance = self.fps * failure_tolerance_ratio
        duration_threshold = self.fps * duration_threshold_in_seconds
        silence_counter = 0
        failure_counter = 0
        intervals_to_cut = []
        absolute_signal = np.absolute(self.signal)
        
        for i, values in tqdm(
            enumerate(absolute_signal),
            desc="Finding silent intervals",
            total=len(absolute_signal),
        ):
            silence = all([value < magnitude_threshold for value in values])
            silence_counter += silence
            failure_counter += not silence
            
            if failure_counter >= failure_tolerance:
                if silence_counter >= duration_threshold:
                    interval_end = (i - failure_counter) / self.fps
                    interval_start = interval_end - (silence_counter / self.fps)

                    interval_start += space_on_edges
                    interval_end -= space_on_edges

                    intervals_to_cut.append(
                        (abs(interval_start), interval_end)
                    )
                silence_counter = 0
                failure_counter = 0
        return intervals_to_cut


class Clip:
    """Video clip processing class for jumpcutting functionality."""
    
    def __init__(self, clip_path: str) -> None:
        self.clip = VideoFileClip(clip_path)
        self.audio = Audio(self.clip.audio)

    def jumpcut(self) -> VideoFileClip:
        """Main jumpcut method that removes silent parts from the video."""
        intervals_to_cut = self.audio.get_intervals_to_cut()
        jumpcutted_clips = []
        previous_stop = 0
        
        for start, stop in tqdm(intervals_to_cut, desc="Removing silent parts"):
            clip_before = self.clip.subclip(previous_stop, start)
            if clip_before.duration > 0.1:  # Keep clips longer than 0.1 seconds
                jumpcutted_clips.append(clip_before)
            previous_stop = stop

        if previous_stop < self.clip.duration:
            last_clip = self.clip.subclip(previous_stop, self.clip.duration)
            jumpcutted_clips.append(last_clip)
            
        return concatenate_videoclips(jumpcutted_clips)


@block_printing
def jumpcutter(input_path: Path) -> None:
    """
    Main jumpcutter function that removes silent parts from videos.
    
    Args:
        input_path: Path to input video file
    """
    print(f"Processing: {input_path}")
    print("Removing silent parts...")
    print(60 * "-")
    
    # Create output filename by adding "_jumpcut" before the extension
    output_path = input_path.parent / f"{input_path.stem}_jumpcut{input_path.suffix}"
    
    clip = Clip(str(input_path))
    jumpcutted_clip = clip.jumpcut()
    
    print(f"Saving to: {output_path}")
    jumpcutted_clip.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac"
    )


def main():
    """Command-line interface for the jumpcutter functionality."""
    if len(sys.argv) != 2:
        print("Usage: python jumpcutter.py <input_video>")
        print("Example: python jumpcutter.py my_video.mp4")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    # Validate input file exists
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)
    
    try:
        jumpcutter(input_path)
        print("Successfully processed video!")
    except Exception as e:
        print(f"Error processing video: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
