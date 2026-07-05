# Real-Time Face Recognition

An educational real-time face-recognition pipeline combining face detection, ArcFace embeddings, cosine-similarity matching, and ByteTrack tracking.

<p align="center">
  <img src="./assets/face-recognition.gif" alt="Real-time face recognition demo" />
</p>

## About this repository

This repository is a study and customization project maintained by [tuanfptu](https://github.com/tuanfptu). It is based on and contains code from [vectornguyen76/face-recognition](https://github.com/vectornguyen76/face-recognition), created by **Vector Nguyễn**.

The original project provided the core architecture and implementation. This version packages that work as a privacy-safe, portable learning repository, with personal face datasets and large pretrained weights excluded from Git.

This is not presented as an original implementation of the underlying face-recognition algorithms.

## Pipeline

```text
Camera / video
      ↓
Face detection (SCRFD, RetinaFace, or YOLOv5-face)
      ↓
Face alignment
      ↓
ArcFace embedding extraction
      ↓
Cosine-similarity identity matching
      ↓
ByteTrack multi-face tracking
```

## Features

- Multiple face detectors: SCRFD, RetinaFace and YOLOv5-face
- ArcFace feature extraction
- Cosine-similarity identity matching
- ByteTrack-based real-time tracking
- Utilities for registering new people
- Webcam and video inference

## Installation

Python 3.9 is recommended because this project uses an older PyTorch-compatible dependency set.

```bash
conda create -n face-dev python=3.9
conda activate face-dev
pip install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Pretrained weights

Large weight files are intentionally excluded from Git. Follow the instructions in:

- `face_detection/scrfd/weights/README.md`
- `face_detection/yolov5_face/weights/README.md`
- `face_recognition/arcface/weights/README.md`
- `face_tracking/pretrained/README.md`

Place downloaded weights in the directories documented by those files.

## Registering a person

Create one directory per person under `datasets/new_persons/`:

```text
datasets/
├── backup/
├── data/
├── face_features/
└── new_persons/
    ├── person_one/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── person_two/
        └── image1.jpg
```

Generate the face database:

```bash
python add_persons.py
```

Start recognition:

```bash
python recognize.py
```

Do not commit face images or generated biometric embeddings. They are excluded through `.gitignore` by default.

## Main technologies

- [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
- [RetinaFace](https://arxiv.org/abs/1905.00641)
- [YOLOv5-face](https://github.com/deepcam-cn/yolov5-face)
- [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)

## Attribution and license

The original implementation is copyright © 2022 Vector Nguyễn and is distributed under the MIT License. In accordance with that license, the original copyright and permission notice remain in [LICENSE.md](./LICENSE.md).

The MIT License permits use, modification, and redistribution, but requires its copyright and permission notice to be retained in copies or substantial portions of the software. Repository ownership on GitHub does not transfer authorship of the original code.

See the original repository: [vectornguyen76/face-recognition](https://github.com/vectornguyen76/face-recognition).
