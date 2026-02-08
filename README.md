FAVencoder (Frame-Accurate Video Encoder) is a comprehensive video processing tool that combines traditional video encoding with AI-powered enhancement capabilities. Built on Python and FFmpeg, it provides both GUI and command-line interfaces for batch processing, frame-accurate editing, and intelligent video upscaling.

![Screenshot 1](https://github.com/minimaster4734/favencoder/blob/main/assets/115506.webp)

 <details>

<summary>Installation</summary>

### Choose the guide for your operating system

Desktop Platforms
- Windows: Full support, including GPU acceleration
- Linux: Full support, with some GPU acceleration limitations
- macOS: Basic support (CPU-only AI enhancement)

Mobile / Terminal Environment
Android (Termux):
- Full core functionality support
- CPU-based AI enhancement available via super-image
- No GPU acceleration
- Standard encoding fully supported

  *ARM Considerations
· GPU AI enhancement disabled on ARM platforms
· CPU AI enhancement available via super-image
· Standard encoding fully supported*

### Core Prerequisites for all systems

  - Python 3.8 or higher.
  - FFmpeg: Must be installed on your system and available in your PATH.

### - Windows/macOS: Tkinter is included with the standard Python installer. No action needed.

### - Linux (for GUI mode): You must install the Tk library and its Python bindings at the system level. This is a prerequisite that cannot be installed via pip.

Step 1: Install the System Graphical Dependency (Linux GUI Users Only)
  ```
  # For Debian/Ubuntu:
  sudo apt update && sudo apt install python3-tk
  # For Fedora/RHEL:
  sudo dnf install python3-tkinter
  ```

Step 2: Set Up an Isolated Python Environment
All subsequent Python packages should be installed inside a virtual environment.

1. Create and activate the environment:
   ```
   python -m venv favenv
   # Activate it:
   # Linux/macOS: source favenv/bin/activate
   # Windows: favenv\Scripts\activate
   ```
   (Your terminal prompt should change to start with (favenv).)
2. Install Core Python Packages via pip:
   With the environment active, use pip to install the latest versions.
   ```
   pip install opencv-python pillow numpy
   ```
   (Do not use sudo or system package managers for this step.)

Step 3: Install Optional AI Packages (Also via pip)
FAVencoder supports AI-powered video upscaling through two backends:

· CPU Backend (Recommended for testing): Install within your active virtual environment.
  ```
  pip install super-image
  ```
· GPU Backend (For faster processing): The application uses Real-ESRGAN-ncnn-vulkan, a standalone executable.
  - No manual installation is required. The program will automatically download the correct version for your operating system the first time you select a GPU AI enhancement option.
  - Requirement: A Vulkan-compatible GPU and drivers. This backend is disabled on ARM-based systems (e.g., Raspberry Pi, Apple Silicon Macs).

  Quick Start Commands

Once your environment is set up and activated:
```
# Launch the graphical interface (default)
python favencoder.py
# Process jobs in the queue without the GUI
python favencoder.py --no-gui
```

### - Android (Termux)
On Termux, all packages can be installed via the pkg manager.
1. Install Termux and All Dependencies
Install Termux from F-Droid. Then, install all dependencies in one step using pkg:
```
# Update the package list and upgrade existing packages
pkg update && pkg upgrade -y

# Install Python, FFmpeg, Tkinter, and all required Python libraries
pkg install python ffmpeg python-tkinter python-numpy python-pillow python-opencv
```

2. Install the AI Package (CPU Only)
The only package you should install via pip is the optional CPU-based AI module. Install it globally.
```
pip install super-image
```
*(You might need to fix dependency issues)*

  Important Notes for Termux:
  - No GPU Acceleration: The Real-ESRGAN GPU backend is not available on Android/Termux (at least not for now)
  - Storage Access: You may need to grant Termux storage permissions (termux-setup-storage) to access video files outside its home directory.

Quick Start in Termux:
```
# Navigate to the directory containing favencoder.py
python favencoder.py
```
</details>
 
 ### Key Features

![Screenshot 2](https://github.com/minimaster4734/favencoder/blob/main/assets/120849.webp)

  🎯 Frame-Accurate Operations
  - Precise Frame Selection: Set exact start and end frames for encoding segments
  - Frame-by-Frame Navigation: Navigate with single-frame precision
  - Timeline Control: Visual timeline with direct frame access
  - Frame-Specific Editing: Apply operations to specific frame ranges

  🎨 Comprehensive Codec Support
  - Video Codecs: FFV1 (Lossless), H.264, H.265, AV1, VP9, ProRes, DNxHD, and hardware-accelerated options (NVENC, QSV, AMF)
  - Audio Codecs: FLAC, PCM, AAC, Opus, MP3, AC3, DTS, Vorbis
  - Custom Encoders: Advanced users can specify any FFmpeg-compatible encoder
  - Intelligent Pairing: Automatic suggestions for optimal video/audio codec combinations

  🔍 Visual Editing Tools
  - Interactive Crop Tool: Click-and-drag cropping with visual handles
  - Real-Time Preview: See crop adjustments immediately
  - Aspect Ratio Maintenance: Intelligent cropping that maintains video proportions
  - Crop History: Save and recall custom crop settings

  🤖 AI-Powered Enhancement
  - Multiple AI Backends: CPU-based (super-image) and GPU-accelerated (Real-ESRGAN)
  - Scale Factors: 2x, 3x, and 4x upscaling
  - Model Specialization: Anime-optimized and general-purpose models

  📁 Batch Processing
- Queue Management: Add multiple videos with consistent settings
- Queue Persistence: Jobs saved between sessions
- Priority Control: Reorder jobs in the queue
- Progress Tracking: Real-time status for each job
- Command Preview: View and edit FFmpeg commands before execution

  🖥️ User Interface
  - Multiple Themes: Light, dark, and grey themes
  - Drag-and-Drop: Load videos by dragging files onto the interface
  - Keyboard Shortcuts: Quick access to common functions
  - Context Menus: Right-click support for text fields
  - Real-Time Updates: Live preview of output settings

### Detailed Feature Guide

  Video Loading & Preview
1. Multiple Loading Methods:
  - File dialog (single or multiple files)
  - Folder loading (process all videos in a folder)
2. Preview Features:
  - Smooth playback with frame-accurate seeking
  - Display of original and output resolutions
  - Aspect ratio information
  - Duration and frame count display

  Crop Tool
1. Activation: Click "Crop Tool" button
2. Usage:
  - Click and drag to create initial selection
  - Resize using corner handles
  - Move by dragging inside the selection
  - Clear with "Clear Crop" button
3. Features:
- Even-dimension enforcement (required by most codecs)
- Aspect ratio display
- Coordinate display

  AI Enhancement
1. Backend Options:
 - CPU Mode: Uses super-image library
 - GPU Mode: Uses Real-ESRGAN with Vulkan acceleration
 - Auto-download: GPU backend downloads automatically if not available
2. Scale Options:
 - 2x, 3x, 4x upscaling
Anime-optimized models
 - General-purpose models (4x only)
3. Processing:
 - Extracts frames to temporary directory
 - Processes each frame with selected AI model
 - Reassembles enhanced frames into video
 - Applies final encoding settings

  Queue System
1. Adding Jobs:
 - Current settings are saved with each job
 - Batch addition from folder loading
 - Individual job editing
2. Queue Management:
 - Reorder jobs with up/down buttons
 - Remove individual jobs
 - Clear entire queue
 - Save/load queue between sessions
3. Processing:
  - Sequential job processing
  - Pause/resume support
  - Stop at any time
  - Progress tracking per job

  Preset System
1. Save Presets: Store current video, audio, and output settings
2. Load Presets: Apply saved settings to current session
3. Preset Files: Stored in JSON format for easy sharing/backup

  Configuration Files

  Queue File (favencoder_queue.json)
 - Format: JSON with job definitions
 - Persistence: Saved automatically after queue modifications
 - Contents: All job parameters including paths, settings, and status

  Preset File (favencoder_presets.json)
 - Format: JSON with preset definitions
 - Manual editing: Possible for advanced users
 - Sharing: Can be copied between installations

  Codec-Specific Features

  Video Codecs
 - FFV1: True lossless encoding with FLAC audio pairing
 - H.264/H.265: Standard compression with quality/bitrate options
 - AV1: Modern compression with SVT-AV1 and AOM implementations
 - Hardware Accelerated: NVENC (NVIDIA), QSV (Intel), AMF (AMD)
 - ProRes/DNxHD: Professional editing formats
 - Custom: Any FFmpeg-compatible encoder

  Audio Codecs
 - Lossless: FLAC, PCM (16/24/32-bit)
 - Lossy: AAC, Opus, MP3, AC3, DTS, Vorbis
 - Custom: Any FFmpeg-compatible audio encoder

  Resolution Options
 - Original resolution
 - Standard definitions (240p to 8K)
 - Custom width/height
 - Custom single dimension (width or height)

  AI Enhancement Resolutions
 - CPU: 2x, 3x, 4x using super-image
 - GPU: 2x, 3x, 4x using Real-ESRGAN
 - GPU Model Types: Anime-optimized and general-purpose

<details>

<summary>Advanced Usage</summary>

  Custom Encoder Arguments
For advanced users who need specific FFmpeg options:
1. Select "Custom (Advanced)" for video or audio codec
2. Enter encoder name (e.g., "libx264")
3. Add additional arguments as needed
4. Supports copy-paste of full FFmpeg command segments

  Output Format Control
· Standard containers: MKV, MP4, MOV, AVI, WebM, FLV, TS
· Custom extensions: Any FFmpeg-supported format
· Audio-only output: When using "No video" codec

  Quality Settings
· CQ (Constant Quality): 0-51 scale (lower = better quality)
· Bitrate: Kilobits per second with VBR/CBR options
· Encoder Speed: Codec-specific presets (ultrafast to placebo)
</details>

<details>

<summary>Technical Advantages</summary> 

  🔄 Minimal Maintenance
· Dependency Light: Only Python and FFmpeg as core dependencies
· System Integration: Leverages system FFmpeg updates
· No Version Lock: Works with any FFmpeg version
· Future-Proof: Core functionality independent of library versions

  🏗️ Modern Architecture
· Type Hints: Full Python type annotations for better code maintenance
· Data Classes: Structured configuration objects
· Enum Usage: Type-safe configuration options
· Separation of Concerns: Clear division between UI, processing, and configuration

  🚀 Performance Features
· Frame Caching: LRU cache for efficient frame retrieval
· Threaded Playback: Smooth preview during processing
· Memory Management: Efficient handling of large videos
· Temp File Cleanup: Automatic cleanup of intermediate files

  🔧 Extensibility
· Modular Design: Easy to add new codecs or features
· Plugin-like Architecture: AI backends can be added or replaced
· Configuration Driven: Settings stored in serializable objects
· API-like Structure: Clear interfaces between components
</details>


 <details>

<summary>Troubleshooting</summary> 

  Common Issues
1. FFmpeg not found: Install FFmpeg and ensure it's in PATH
2. GPU acceleration not working: Check Vulkan compatibility (not available on Termux)
3. AI enhancement very slow: Consider GPU acceleration or lower scale factor (on Termux, only CPU is available)
4. Termux package issues: Use 'pkg install' for core packages, not pip

  Logging
Enable detailed logging by setting ENABLE_LOGGING = True at the top of the favencoder.py script. Logs are written to favencoder.log.
</details>

 
 <details>

<summary>Development Notes</summary>

  Code Structure
· Main Classes: VideoCropper (main app), VideoPlayer, ConversionJob
· Data Classes: VideoSettings, AudioSettings, OutputSettings, CropRect
· Enums: VideoCodec, AudioCodec, ResolutionMode, QualityMode
· Utilities: LRUCache, format conversion functions

  In case you want to add new features by yourself:
1. New Codec: Add to appropriate Enum class and update encoder mapping
2. New AI Backend: Implement enhancement method and add to availability check
3. New UI Feature: Add to appropriate frame in build_ui methods

  Performance Tips
1. For large videos: Use frame range selection to process only needed segments
2. For batch processing: Use queue system with consistent settings
3. For AI enhancement: GPU acceleration provides 10-100x speed improvement (not available on Termux)
4. For preview: Lower resolution videos load and process faster

  License & Attribution

  This tool is built on:
· FFmpeg: Video processing backbone
· Real-ESRGAN: AI upscaling (when GPU acceleration used)
· super-image: CPU-based AI upscaling
· OpenCV: Frame extraction and processing
· Pillow: Image manipulation

  Support
  For issues, feature requests, or contributions:
1. Check existing documentation
2. Review console/log output
3. Ensure FFmpeg is properly installed
4. Test with a small video file first
5. For Termux issues: Ensure all packages were installed via pkg as instructed
</details>

