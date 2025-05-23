# Video Annotation Tool

A python tool for annotating videos with custom labels and timestamps. Designed for researchers and developers who need to tag video content efficiently.

![image](https://github.com/user-attachments/assets/0ae8d2cc-935c-41a1-b661-a15fbfabce91)

## Features

- Add annotations with timestamps every set number of frames
- Export annotations or CSV and video
- User-friendly interface

## Getting Started

### Prerequisites

- OpenCV

## Usage

1. Modify the following attributes in ``main.py`` accordingly:
    * VIDEO_FILE: Path to the video to annotate.
    * NUMBER_OF_ELEMENTS: Number of elements to annotate.
    * FRAME_SKIP: Number of frames between each annotation.
    * OUTPUT_FOLDER: Path where the files will be stored. (New folder will be created to store all outputs)
    * ANNOTATION_WINDOW_WIDTH: Width of the window to annotate. (Output will have the same resolution as input)
2. Run ``main.py``.
3. Add annotations sequentially for each frame by:
    * Left clicking and dragging to create a square
    * Right clicking to skip an element for that frame
4. Output will be automatically exported as a mp4 video showing the annotations and a csv will the annotations for all frames.

## Suggested File Structure

```
Video-Annotation-Tool/
├── videos/
│   └── video1.mp4
│   └── video2.mp4
│   └── ...
├── output/
├── classes/
└── main.py
```
