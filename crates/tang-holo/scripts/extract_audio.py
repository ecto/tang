#!/usr/bin/env python3
"""Extract audio features from a video file for talking head animation.

Extracts per-frame audio features:
- 80-dim log mel spectrogram (25ms window, 10ms hop)
- Resampled to match video FPS

Usage:
    python extract_audio.py <video_path> <fps> <output.bin>

Output format (binary):
    - u32: num_frames
    - u32: feature_dim (80)
    - u32: sample_rate (16000)
    - For each frame:
        - [f32; 80]: mel spectrogram features
"""

import sys
import struct
import numpy as np
from pathlib import Path

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <video_path> <fps> <output.bin>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    fps = float(sys.argv[2])
    output_path = Path(sys.argv[3])

    # Extract audio using ffmpeg
    import subprocess
    import tempfile

    wav_path = Path(tempfile.mktemp(suffix='.wav'))

    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', str(video_path),
            '-ar', '16000', '-ac', '1', '-f', 'wav',
            str(wav_path)
        ], check=True, capture_output=True)
    except FileNotFoundError:
        print("ffmpeg not found")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr.decode()}")
        sys.exit(1)

    # Load audio
    try:
        import scipy.io.wavfile as wavfile
        import scipy.signal
    except ImportError:
        print("Install: pip install scipy")
        sys.exit(1)

    sr, audio = wavfile.read(str(wav_path))
    wav_path.unlink()  # cleanup

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    print(f"Audio: {len(audio)} samples @ {sr}Hz ({len(audio)/sr:.1f}s)")

    # Compute log mel spectrogram
    n_fft = 400      # 25ms @ 16kHz
    hop_length = 160  # 10ms @ 16kHz
    n_mels = 80

    # Mel filterbank
    mel_basis = mel_filterbank(sr, n_fft, n_mels)

    # STFT
    window = np.hanning(n_fft)
    pad_length = n_fft // 2
    audio_padded = np.pad(audio, (pad_length, pad_length), mode='reflect')

    num_spec_frames = 1 + (len(audio_padded) - n_fft) // hop_length
    mel_features = np.zeros((num_spec_frames, n_mels), dtype=np.float32)

    for i in range(num_spec_frames):
        start = i * hop_length
        frame = audio_padded[start:start + n_fft] * window
        spectrum = np.abs(np.fft.rfft(frame)) ** 2
        mel = mel_basis @ spectrum
        mel_features[i] = np.log(np.maximum(mel, 1e-10))

    print(f"Mel spectrogram: {mel_features.shape}")

    # Resample to video FPS
    spec_fps = sr / hop_length  # typically 100 fps
    num_video_frames = int(len(audio) / sr * fps)

    video_features = np.zeros((num_video_frames, n_mels), dtype=np.float32)
    for i in range(num_video_frames):
        t = i / fps  # time in seconds
        spec_idx = int(t * spec_fps)
        spec_idx = min(spec_idx, num_spec_frames - 1)
        video_features[i] = mel_features[spec_idx]

    print(f"Video-aligned features: {video_features.shape} ({num_video_frames} frames @ {fps} fps)")

    # Write output
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<III', num_video_frames, n_mels, sr))
        f.write(video_features.tobytes())

    print(f"Done! -> {output_path}")

def mel_filterbank(sr, n_fft, n_mels, fmin=0, fmax=None):
    """Compute mel filterbank matrix."""
    if fmax is None:
        fmax = sr / 2

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    n_freq = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freq))

    for m in range(n_mels):
        f_left = bin_points[m]
        f_center = bin_points[m + 1]
        f_right = bin_points[m + 2]

        for k in range(f_left, f_center):
            if f_center != f_left:
                filterbank[m, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right != f_center:
                filterbank[m, k] = (f_right - k) / (f_right - f_center)

    return filterbank

if __name__ == '__main__':
    main()
