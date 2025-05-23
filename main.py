import os

from classes.annotator import Annotator

VIDEO_FILE = 'videos/short clips/ENTRY_16.mp4'
NUMBER_OF_ELEMENTS = 1
FRAME_SKIP = 5
OUTPUT_FOLDER = 'output'
ANNOTATION_WINDOW_WIDTH = 1000

if __name__ == "__main__":
    video = VIDEO_FILE[VIDEO_FILE.rfind('/') + 1:]
    if not os.path.exists(os.path.join(OUTPUT_FOLDER, video)):
        os.makedirs(os.path.join(OUTPUT_FOLDER, video))

    annotator = Annotator(VIDEO_FILE, NUMBER_OF_ELEMENTS, FRAME_SKIP, os.path.join(OUTPUT_FOLDER, video), ANNOTATION_WINDOW_WIDTH)
    annotator.run()