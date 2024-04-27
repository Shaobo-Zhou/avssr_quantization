import argparse
import os
import subprocess
from pathlib import Path

from tqdm.auto import tqdm

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent


def get_audio_from_video(dataset_name, video_root_path):
    meta_path = ROOT_PATH / "data" / dataset_name

    # Setting audio Parameters
    sr = 16000  # sample rate
    # start_time = 0.0 # cut start time
    # length_time = 2.0 # cut audio length
    outpath = str(meta_path) + "/long_raw_audio"
    text_outpath = str(meta_path) + "/raw_text"
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(text_outpath, exist_ok=True)

    train_mouth = []
    val_mouth = []
    test_mouth = []
    train = open(str(meta_path) + "/retrain.txt", "r").readlines()
    test = open(str(meta_path) + "/retest.txt", "r").readlines()
    val = open(str(meta_path) + "/redev.txt", "r").readlines()

    for filename in tqdm(train):
        filename = filename.replace("\n", "")
        train_mouth.append(filename)

    for filename in tqdm(val):
        filename = filename.replace("\n", "")
        val_mouth.append(filename)

    for filename in tqdm(test):
        filename = filename.replace("\n", "")
        test_mouth.append(filename)

    with open(str(meta_path) + "/video_path.txt", "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            if line != "":
                line = line.replace("\n", "")
                line = video_root_path + "/" + line[23:]  # fix video_path
                filename = line.split("/")[-2] + "_" + line.split("/")[-1].split(".")[0]

                text_line = line[:-3] + "txt"
                with open(text_line, "r") as text_f:
                    text = text_f.readline().strip()
                    text = text[5:]  # remove TEXT:
                    text = text.strip()

                with open(text_outpath + "/" + filename + ".txt", "w") as text_f:
                    text_f.write(text + "\n")

                if filename in train_mouth:
                    path = outpath + "/train"
                    os.makedirs(path, exist_ok=True)
                    command = ""
                    command += "ffmpeg -i {} -f wav -ar {} -ac 1 {}/{}.wav;".format(
                        line, sr, path, filename
                    )
                    # command += 'ffmpeg -i {} -f wav -ar {} -ac 1 {}/tmp_{}.wav;'.format(line, sr, path, l)
                    # command += 'sox {}/tmp_{}.wav {}/{}.wav trim {} {};'.format(path, l, path, l, start_time, length_time)
                    # command += 'rm {}/tmp_{}.wav;'.format(path, l)
                    p = subprocess.Popen(
                        command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    p.wait()
                if filename in test_mouth:
                    path = outpath + "/test"
                    os.makedirs(path, exist_ok=True)
                    command = ""
                    command += "ffmpeg -i {} -f wav -ar {} -ac 1 {}/{}.wav;".format(
                        line, sr, path, filename
                    )
                    # command += 'ffmpeg -i {} -f wav -ar {} -ac 1 {}/tmp_{}.wav;'.format(line, sr, path, l)
                    # command += 'sox {}/tmp_{}.wav {}/{}.wav trim {} {};'.format(path, l, path, l, start_time, length_time)
                    # command += 'rm {}/tmp_{}.wav;'.format(path, l)
                    p = subprocess.Popen(
                        command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    p.wait()
                if filename in val_mouth:
                    path = outpath + "/val"
                    os.makedirs(path, exist_ok=True)
                    command = ""
                    command += "ffmpeg -i {} -f wav -ar {} -ac 1 {}/{}.wav;".format(
                        line, sr, path, filename
                    )
                    # command += 'ffmpeg -i {} -f wav -ar {} -ac 1 {}/tmp_{}.wav;'.format(line, sr, path, l)
                    # command += 'sox {}/tmp_{}.wav {}/{}.wav trim {} {};'.format(path, l, path, l, start_time, length_time)
                    # command += 'rm {}/tmp_{}.wav;'.format(path, l)
                    p = subprocess.Popen(
                        command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    p.wait()
            else:
                pass


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Get raw audio and text from video")
    args.add_argument(
        "-d",
        "--dataset_name",
        default=None,
        type=str,
        help="Dataset name inside data dir (default: None)",
    )

    args.add_argument(
        "-v",
        "--video_root_path",
        default=None,
        type=str,
        help="Path to raw video dir where mvlrs dir is saved (default: None)",
    )

    args = args.parse_args()

    assert args.video_root_path is not None, "Provide the path to root of mvlrs dir"

    get_audio_from_video(args.dataset_name, args.video_root_path)
