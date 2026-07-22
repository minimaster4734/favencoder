### FAVencoder
FAVencoder (Frames, Audio & Video Encoder) is a Python + FFmpeg video processing tool focused on frame-accurate editing, batch encoding, and AI-powered upscaling.
It offers both a graphical interface and a command-line workflow, making it suitable for everything from quick edits to large batch jobs.

Runs on Windows, Linux, macOS, and Android. Works on x86, ARM, and RISC-V architectures (see "Installation > Architecture Support" for details).

![Screenshot 1](https://github.com/minimaster4734/favencoder/blob/main/assets/163798.png)

<details>
<summary>Installation</summary>

### Core Prerequisites for all systems

- Python 3.8 or higher
- FFmpeg: must be installed on your system and available in your PATH

### Windows/macOS: Tkinter is included with the standard Python installer — no action needed.

### Linux (GUI mode only): the Tk library and its Python bindings must be installed at the system level; this can't be installed via pip.

Step 1: Install the System Graphical Dependency (Linux GUI users only)
```
# Debian/Ubuntu:
sudo apt update && sudo apt install python3-tk
# Fedora/RHEL:
sudo dnf install python3-tkinter
```

Step 2: Set Up an Isolated Python Environment

All Python packages below should be installed inside a virtual environment.
```
python -m venv favenv
# Activate it:
# Linux/macOS: source favenv/bin/activate
# Windows: favenv\Scripts\activate
```
(Your terminal prompt should change to start with `(favenv)`.)

Install the core packages:
```
pip install opencv-python pillow numpy requests
```
> Running `--no-gui` only? (See Quick Start below) These four packages aren't loaded at all, so a CLI-only/headless install can skip them until you need that feature.

Step 3: Install Optional AI Packages

FAVencoder supports AI-powered video upscaling through two backends:
- **CPU backend:** Uses waifu2x, inside your active virtual environment.
- **GPU backend:** uses Real-ESRGAN-ncnn-vulkan, a standalone executable. No manual install needed — it's downloaded automatically the first time you select a GPU AI enhancement option. Requires a Vulkan-compatible GPU and drivers; unavailable on ARM (see Architecture Support below).

### Android (Termux) Setup

Install [Termux from F-Droid](https://f-droid.org/), then install everything in one step with `pkg`:
```
# Update packages
pkg update && pkg upgrade -y

# Install Python, FFmpeg, Tkinter, and required Python libraries
pkg install python ffmpeg python-tkinter python-numpy python-pillow python-opencv && pip install requests
```
(Optional) Install the CPU AI package globally: `pip install waifu2x` (you might need to fix dependency issues).

**Notes for Termux:**
- No GPU acceleration for now (see Architecture Support above).
- You may need to grant storage permissions (`termux-setup-storage`) to access video files outside Termux's home directory.

### Architecture Support

- **x86 (64-bit):** Full support, including GPU acceleration.
- **ARM64** (Raspberry Pi, Apple Silicon, Android/Termux): Full core functionality and CPU AI enhancement via `waifu2x`. GPU AI acceleration is not available.
- **RISC-V 64:** Core functionality works once the required Python packages are installed — use **conda-forge** (pre-built `riscv64` packages) or wheels from the [RISE project](https://github.com/riscv-forks/riscv-wheels). GPU acceleration is unavailable; CPU AI enhancement may work if `waifu2x` installs.


### Quick Start
```
'cd' into the folder where favencoder.py is.
# Launch the graphical interface (default)
python favencoder.py OR python3 favencoder.py
# Process jobs in the queue without the GUI
python favencoder.py --no-gui OR python3 favencoder.py --no-gui

</details>

### Key Features Overview

- Frame-accurate editing (exact start/end frames, frame-by-frame navigation)
- Batch processing with persistent queues
- Interactive visual crop tool
- AI upscaling (CPU & GPU backends)
- Wide codec support (lossless, modern, and hardware-accelerated)
- GUI + CLI modes

![Screenshot 2](https://github.com/minimaster4734/favencoder/blob/main/assets/120849.webp)

<details>
<summary>Full Feature List</summary>

**Frame-Accurate Operations**
- Precise frame selection: set exact start and end frames for encoding segments
- Frame-by-frame navigation with single-frame precision
- Visual timeline with direct frame access
- Apply operations to specific frame ranges

**Comprehensive Codec Support**
- Video: FFV1 (lossless, paired with FLAC audio), H.264, H.265, AV1 (SVT-AV1 and AOM implementations), VP9, ProRes, DNxHD, VVC (if available on FFMPEG build) and hardware-accelerated options (NVENC, QSV, AMF)
- Audio: FLAC, PCM, AAC, Opus, MP3, AC3, DTS, Vorbis
- Custom encoders: advanced users can specify any FFmpeg-compatible encoder
- Intelligent pairing: automatic suggestions for optimal video/audio codec combinations
- Resolution options: original, standard presets (240p–8K), custom width/height, or a single custom dimension

**Visual Editing Tools**
- Interactive crop tool: click-and-drag cropping with visual handles and real-time preview
- Aspect-ratio-aware, even-dimension cropping (required by most codecs)
- Save and recall custom crop settings

**AI-Powered Enhancement**
- CPU backend (`waifu2x`) and GPU-accelerated backend (Real-ESRGAN); GPU backend downloads automatically on first use
- Scale factors: 2x, 3x, 4x
- Anime-optimized and general-purpose models (general-purpose is 4x only)
- Pipeline: extracts frames to a temp directory → enhances each frame → reassembles into video → applies final encoding settings

**Batch Processing**
- Queue with persistence across sessions; each job stores the settings it was added with
- Reorder, remove, or clear jobs; sequential processing with pause/resume/stop and per-job progress
- Command preview: view and edit the FFmpeg command before running it

**User Interface**
- Light, dark, and grey themes
- Keyboard shortcuts and right-click context menus on text fields
- Live preview of output settings

### Detailed Feature Guide

**Video Loading & Preview**
- Load via file dialog (single or multiple files) or by loading a whole folder
- Smooth, frame-accurate seeking; shows original/output resolution, aspect ratio, duration, and frame count

**Crop Tool**
1. Click "Crop Tool" to activate.
2. Click and drag to create a selection, resize with the corner handles, move by dragging inside it, or clear with "Clear Crop."

**Preset System**
- Save current video, audio, and output settings as a preset; load a preset to apply it to the current session
- Presets are stored as JSON, so they're easy to share or back up

### Configuration Files

**Queue file** (`favencoder_queue.json`): JSON job definitions — paths, settings, and status — saved automatically after any queue change.
**Preset file** (`favencoder_presets.json`): JSON preset definitions; editable by hand if needed, and portable between installs.

</details>

<details>
<summary>Advanced Usage</summary>

**Custom Encoder Arguments**

For specific FFmpeg options:
1. Select "Custom (Advanced)" for the video or audio codec.
2. Enter the encoder name (e.g., `libx264`).
3. Add any additional arguments — full FFmpeg command segments can be pasted directly.

**Output Format Control**
- Standard containers: MKV, MP4, MOV, AVI, WebM, FLV, TS
- Any FFmpeg-supported custom extension
- Audio-only output when using the "No video" codec

**Quality Settings**
- CQ (Constant Quality): 0–51 scale, lower is better quality
- Bitrate: kbps, with VBR/CBR options
- Encoder speed: codec-specific presets (ultrafast → placebo)

</details>

<details>
<summary>Technical Advantages</summary>

**Minimal Footprint**
- No bundled ML frameworks or vendored binaries — Real-ESRGAN/waifu2x are fetched on demand, and everything else layers on your system's own FFmpeg install
- No version lock: works with any FFmpeg version
- Core functionality is independent of any single library's release cycle

**Modern Architecture**
- Full type hints throughout
- Dataclasses for structured configuration
- Enums for type-safe configuration options
- Clear separation between UI, processing, and configuration

**Performance**
- LRU frame cache for efficient retrieval
- Threaded playback for a smooth preview during processing
- Automatic cleanup of temp files and intermediate frames

**Extensibility**
- Modular design — new codecs or AI backends can be added without touching unrelated code
- Settings are fully serializable, so presets/queues/config are all plain JSON

</details>

<details>
<summary>Troubleshooting</summary>

**Common Issues**
1. FFmpeg not found: install FFmpeg and make sure it's on your PATH.
2. GPU acceleration not working: check Vulkan compatibility (not available on Termux/ARM).
3. AI enhancement very slow: use GPU acceleration if available, or a lower scale factor.
4. Termux package issues: use `pkg install` for core packages, not pip.

**Logging**

Set `ENABLE_LOGGING = True` at the top of `favencoder.py` for detailed logs, written to `favencoder.log`.

</details>

<details>
<summary>Development Notes</summary>

**Code Structure**
- Main classes: `VideoCropper` (main app), `VideoPlayer`, `ConversionJob`
- Dataclasses: `VideoSettings`, `AudioSettings`, `OutputSettings`, `CropRect`
- Enums: `VideoCodec`, `AudioCodec`, `ResolutionMode`, `QualityMode`
- Utilities: `LRUCache`, format conversion helpers

**Adding new features**
1. New codec: add it to the relevant Enum and update the encoder mapping.
2. New AI backend: implement an enhancement method and add it to the availability check.
3. New UI feature: add it to the relevant frame in the `build_ui` methods.

**Performance Tips**
1. Use frame-range selection to process only the segment you need on large videos.
2. Use the queue system for consistent settings across a batch.
3. GPU acceleration gives roughly 10–100x speedup for AI enhancement where available.
4. Lower-resolution source video loads and previews faster.

**License & Attribution**

Built on:
- FFmpeg — video processing backbone
- Real-ESRGAN — GPU-accelerated AI upscaling
- waifu2x — CPU-based AI upscaling
- OpenCV — frame extraction and processing
- Pillow — image manipulation

**Support**

For issues, feature requests, or contributions:
1. Check the existing documentation.
2. Review console/log output.
3. Confirm FFmpeg is properly installed.
4. Test with a small video file first.
5. Termux-specific issues: confirm all packages were installed via `pkg`, as instructed above.

</details>
