# tang reel

A 47-second, beat-synced hype reel rendered from the site's own demos. Every
number and graph on screen comes from tang running in WASM (`site/pkg/tang`)
or from `site/graphs.json`, the kernel graph exported from the tensor IR.

The soundtrack is not in the repo. Bring a track you have the rights to; the
scripts below find its tempo, downbeats and drops.

## Steps

```bash
cd tools/reel
npm install
mkdir -p audio

# 1. audio: a stereo wav for muxing, a mono 11 kHz copy for analysis
ffmpeg -i /path/to/track -vn -ac 2 -ar 48000 audio/track.wav
ffmpeg -i audio/track.wav -ac 1 -ar 11025 -f s16le audio/mono.raw

# 2. tempo and beat grid, then downbeat phase and candidate drops
python3 beats.py audio/mono.raw audio/beats.json
python3 drop.py audio/mono.raw audio/beats.json

# 3. soundtrack: intro from 0, then the two drops, each cut on a downbeat
#    (override INTRO_END / DROP1 / DROP2 with the times from step 2)
./splice.sh audio/track.wav

# 4. frames: serve the repo root, then render 1413 frames at 30 fps
(cd ../.. && python3 -m http.server 8766) &
node render.mjs 1920 1080 frames

# 5. mux
ffmpeg -framerate 30 -i frames/%05d.jpg -i audio/spliced.wav \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -c:a aac -b:a 192k \
  -shortest -movflags +faststart tang-reel.mp4
```

Open `http://localhost:8766/tools/reel/reel.html?fit=1` and call
`renderAt(seconds)` in the console to scrub a single frame.

## Timeline

Beat 0 is the first downbeat (`OFFSET` in `reel.html`, 0.862 s at 135 BPM).
Scenes start on beats and every reveal lands on a beat or a subdivision.

| beats | section | scenes |
|---|---|---|
| 0–16 | intro, track from 0 | logo slam, "Write math in Rust.", muni logo flash, "Compile it." with a 3-2-1 |
| 16–40 | drop 1 | "Compile it to …" and the backends; dual numbers, the graph, the kernel source flipping dialects, 47 → 17 kernels |
| 40–64 | drop 1, performance | 3.5× vs PyTorch, 7.6× derived flash attention, 122 tok/s gauge, 64/64 tokens, +18% / 0 lines, 6 backends |
| 64–88 | drop 2, use cases | eight scenes, three beats each |
| 88–104 | close | `cargo add tang`, muni end card |

The muni logo sits in the corner from beat 16 to 88, as in the BVR1 reel.

Chrome is found at the default macOS path; set `CHROME` to use another.
