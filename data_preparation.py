import cv2
import os
import pandas as pd
import numpy as np 
import skvideo.io

ROOT_DIR = "./"

DATA_DIR = os.path.join(ROOT_DIR, "data/")

TRAIN_VIDEO = os.path.join(DATA_DIR, "train.mp4")
TEST_VIDEO = os.path.join(DATA_DIR, "test.mp4")

TRAIN_FRAME_DIR = os.path.join(DATA_DIR, "train_frames/")
TEST_FRAME_DIR = os.path.join(DATA_DIR, "test_frames/")

TRAIN_LABELS = os.path.join(DATA_DIR, "train.txt")
TRAIN_METADATA = os.path.join(DATA_DIR, "train.csv")

TEST_METADATA = os.path.join(DATA_DIR, "test.csv")


def generate_frames_opencv(video, output_dir):
    FPS = 20
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except OSError:
        print('Error: Creating directory')

    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        name = output_dir + str(current_frame) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
        current_frame += 1
    cap.release()
    cv2.destroyAllWindows()


def generate_frames_skv(video, output_dir):
    reader = skvideo.io.FFmpegReader(video)

    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except OSError:
        print('Error: Creating directory')

    for idx, frame in enumerate(reader.nextFrame()):
        img_path = output_dir + str(idx) + '.jpg'
        print('Creating...', img_path)
        skvideo.io.vwrite(img_path, frame)


def create_train_dataframe(label_file, metadata_file):
    data = pd.DataFrame(pd.read_csv(label_file,
                                    header=None,
                                    squeeze=True))
    data.columns = ["speed"]
    data["img_index"] = data.index
    data["file_name"] = data.img_index.apply(lambda x: str(x) + ".jpg")
    data.to_csv(metadata_file, index=False)


def create_test_dataframe(frame_dir, metadata_file):
    frame_count = len(os.listdir(frame_dir))
    data = pd.DataFrame(list(range(frame_count)))
    data.columns = ["img_index"]
    data["file_name"] = data.img_index.apply(lambda x: str(x) + ".jpg")
    data["speed"] = None
    data.to_csv(metadata_file, index=False)


if __name__ == "__main__":
    # generate_frames_opencv(TRAIN_VIDEO, TRAIN_FRAME_DIR)
    # generate_frames_opencv(TEST_VIDEO, TEST_FRAME_DIR)

    generate_frames_skv(TRAIN_VIDEO, TRAIN_FRAME_DIR)
    generate_frames_skv(TEST_VIDEO, TEST_FRAME_DIR)

    create_train_dataframe(TRAIN_LABELS, TRAIN_METADATA)
    create_test_dataframe(TEST_FRAME_DIR, TEST_METADATA)











