"""
Standalone Jumpcutter - Remove Silent Parts from Videos

A complete, self-contained implementation for removing silent parts from videos.
This module provides functionality to:
- Detect silent segments in video audio
- Remove or speed up silent parts
- Generate jumpcut videos with configurable parameters

Dependencies:
- moviepy
- numpy
- tqdm

Usage:
    python standalone_jumpcutter.py input_video.mp4 output_video.mp4
    python standalone_jumpcutter.py input_video.mp4 output_video.mp4 --magnitude-threshold 0.02 --duration-threshold 1.0
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.fx.all import speedx
from moviepy.video.io.VideoFileClip import VideoFileClip
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

    def get_intervals_to_cut(
        self,
        magnitude_threshold_ratio: float,
        duration_threshold_in_seconds: float,
        failure_tolerance_ratio: float,
        space_on_edges: float,
    ) -> List[Tuple[float, float]]:
        """
        Analyze audio to find intervals that should be cut (silent parts).
        
        Args:
            magnitude_threshold_ratio: Ratio for determining silence threshold
            duration_threshold_in_seconds: Minimum duration of silence to cut
            failure_tolerance_ratio: Tolerance for non-silent samples
            space_on_edges: Buffer time around silence intervals
            
        Returns:
            List of (start, end) tuples representing intervals to cut
        """
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
            desc="Getting silent intervals to cut",
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
                    )  # in seconds
                silence_counter = 0
                failure_counter = 0
        return intervals_to_cut


class Clip:
    """
    Video clip processing class for jumpcutting functionality.
    """
    
    def __init__(
        self, clip_path: str, min_loud_part_duration: int, silence_part_speed: int
    ) -> None:
        self.clip = VideoFileClip(clip_path)
        self.audio = Audio(self.clip.audio)
        self.cut_to_method = {
            "silent": self.jumpcut_silent_parts,
            "voiced": self.jumpcut_voiced_parts,
        }
        self.min_loud_part_duration = min_loud_part_duration
        self.silence_part_speed = silence_part_speed

    def jumpcut(
        self,
        cuts: List[str],
        magnitude_threshold_ratio: float,
        duration_threshold_in_seconds: float,
        failure_tolerance_ratio: float,
        space_on_edges: float,
    ) -> Dict[str, VideoFileClip]:
        """
        Main jumpcut method that processes the video based on audio analysis.
        
        Args:
            cuts: List of cut types ("silent" or "voiced")
            magnitude_threshold_ratio: Audio magnitude threshold for silence detection
            duration_threshold_in_seconds: Minimum silence duration to cut
            failure_tolerance_ratio: Tolerance for non-silent samples
            space_on_edges: Buffer time around silence intervals
            
        Returns:
            Dictionary mapping cut types to processed video clips
        """
        intervals_to_cut = self.audio.get_intervals_to_cut(
            magnitude_threshold_ratio,
            duration_threshold_in_seconds,
            failure_tolerance_ratio,
            space_on_edges,
        )
        outputs = {}
        for cut in cuts:
            jumpcutted_clips = self.cut_to_method[cut](intervals_to_cut)
            outputs[cut] = concatenate_videoclips(jumpcutted_clips)
        return outputs

    def jumpcut_silent_parts(
        self, intervals_to_cut: List[Tuple[float, float]]
    ) -> List[VideoFileClip]:
        """
        Remove silent parts from the video, optionally speeding them up.
        
        Args:
            intervals_to_cut: List of (start, end) tuples for silent intervals
            
        Returns:
            List of video clips with silent parts removed/speeded up
        """
        jumpcutted_clips = []
        previous_stop = 0
        
        for start, stop in tqdm(intervals_to_cut, desc="Cutting silent intervals"):
            clip_before = self.clip.subclip(previous_stop, start)

            if clip_before.duration > self.min_loud_part_duration:
                jumpcutted_clips.append(clip_before)

            if self.silence_part_speed is not None:
                silence_clip = self.clip.subclip(start, stop)
                silence_clip = speedx(
                    silence_clip, self.silence_part_speed
                ).without_audio()
                jumpcutted_clips.append(silence_clip)

            previous_stop = stop

        if previous_stop < self.clip.duration:
            last_clip = self.clip.subclip(previous_stop, self.clip.duration)
            jumpcutted_clips.append(last_clip)
        return jumpcutted_clips

    def jumpcut_voiced_parts(
        self, intervals_to_cut: List[Tuple[float, float]]
    ) -> List[VideoFileClip]:
        """
        Keep only the voiced parts (opposite of silent parts).
        
        Args:
            intervals_to_cut: List of (start, end) tuples for silent intervals
            
        Returns:
            List of video clips containing only voiced parts
        """
        jumpcutted_clips = []
        for start, stop in tqdm(intervals_to_cut, desc="Cutting voiced intervals"):
            if start < stop:
                silence_clip = self.clip.subclip(start, stop)
                jumpcutted_clips.append(silence_clip)
        return jumpcutted_clips


@block_printing
def jumpcutter(
    input_path: Path, 
    output_path: Path, 
    magnitude_threshold_ratio: float = 0.02, 
    duration_threshold: float = 1.0,
    failure_tolerance_ratio: float = 0.1, 
    space_on_edges: float = 0.2, 
    silence_part_speed: int = None, 
    min_loud_part_duration: int = -1,
    codec: str = "libx264", 
    bitrate: str = None, 
    cut: str = 'silent'
) -> None:
    """
    Main jumpcutter function that removes silent parts from videos.
    
    Args:
        input_path: Path to input video file
        output_path: Path for output video file
        magnitude_threshold_ratio: Audio magnitude threshold for silence detection (default: 0.02)
        duration_threshold: Minimum silence duration to cut in seconds (default: 1.0)
        failure_tolerance_ratio: Tolerance for non-silent samples (default: 0.1)
        space_on_edges: Buffer time around silence intervals in seconds (default: 0.2)
        silence_part_speed: Speed multiplier for silent parts (None = remove, int = speed up)
        min_loud_part_duration: Minimum duration for loud parts to keep (default: -1)
        codec: Video codec for output (default: "libx264")
        bitrate: Video bitrate for output (default: None)
        cut: Type of cutting - "silent", "voiced", or "both" (default: "silent")
    """
    print(f"Processing: {input_path} -> {output_path}")
    print(f"Magnitude threshold: {magnitude_threshold_ratio}")
    print(f"Duration threshold: {duration_threshold} seconds")
    print(60 * "-")
    
    if duration_threshold / 2 <= space_on_edges:
        print(60 * "*")
        print("WARNING:")
        print("You have selected space_on_edges >= duration_threshold/2. This may cause overlapping sequences")
        print(60 * "*")
    
    cuts = [cut] if cut != "both" else ["silent", "voiced"]
    codec = 'libx264'
    bitrate = bitrate

    clip = Clip(str(input_path), min_loud_part_duration, silence_part_speed)
    outputs = clip.jumpcut(
        cuts,
        magnitude_threshold_ratio,
        duration_threshold,
        failure_tolerance_ratio,
        space_on_edges,
    )
    
    for cut_type, jumpcutted_clip in outputs.items():
        if len(outputs) == 2:
            output_filename = output_path.parent / f"{output_path.stem}_{cut_type}_parts_cutted{output_path.suffix}"
            jumpcutted_clip.write_videofile(
                str(output_filename),
                codec=codec,
                bitrate=bitrate,
                audio_codec="aac"
            )
        else:
            jumpcutted_clip.write_videofile(
                str(output_path), 
                codec=codec, 
                bitrate=bitrate, 
                audio_codec="aac"
            )


def main():
    """
    Command-line interface for the jumpcutter functionality.
    """
    parser = argparse.ArgumentParser(
        description="Remove silent parts from videos (jumpcutter)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - remove silence
  python standalone_jumpcutter.py input.mp4 output.mp4
  
  # Custom thresholds
  python standalone_jumpcutter.py input.mp4 output.mp4 --magnitude-threshold 0.01 --duration-threshold 2.0
  
  # Speed up silence instead of removing
  python standalone_jumpcutter.py input.mp4 output.mp4 --silence-speed 3
  
  # Cut both silent and voiced parts
  python standalone_jumpcutter.py input.mp4 output.mp4 --cut both
        """
    )
    
    parser.add_argument("input", type=str, help="Input video file path")
    parser.add_argument("output", type=str, help="Output video file path")
    parser.add_argument("--magnitude-threshold", type=float, default=0.02,
                       help="Audio magnitude threshold for silence detection (default: 0.02)")
    parser.add_argument("--duration-threshold", type=float, default=1.0,
                       help="Minimum silence duration to cut in seconds (default: 1.0)")
    parser.add_argument("--failure-tolerance", type=float, default=0.1,
                       help="Tolerance for non-silent samples (default: 0.1)")
    parser.add_argument("--space-on-edges", type=float, default=0.2,
                       help="Buffer time around silence intervals in seconds (default: 0.2)")
    parser.add_argument("--silence-speed", type=int, default=None,
                       help="Speed multiplier for silent parts (None = remove, int = speed up)")
    parser.add_argument("--min-loud-duration", type=int, default=-1,
                       help="Minimum duration for loud parts to keep (default: -1)")
    parser.add_argument("--codec", type=str, default="libx264",
                       help="Video codec for output (default: libx264)")
    parser.add_argument("--bitrate", type=str, default=None,
                       help="Video bitrate for output (default: None)")
    parser.add_argument("--cut", type=str, choices=["silent", "voiced", "both"], default="silent",
                       help="Type of cutting - silent, voiced, or both (default: silent)")
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        jumpcutter(
            input_path=input_path,
            output_path=output_path,
            magnitude_threshold_ratio=args.magnitude_threshold,
            duration_threshold=args.duration_threshold,
            failure_tolerance_ratio=args.failure_tolerance,
            space_on_edges=args.space_on_edges,
            silence_part_speed=args.silence_speed,
            min_loud_part_duration=args.min_loud_duration,
            codec=args.codec,
            bitrate=args.bitrate,
            cut=args.cut
        )
        print(f"Successfully processed video: {output_path}")
    except Exception as e:
        print(f"Error processing video: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
