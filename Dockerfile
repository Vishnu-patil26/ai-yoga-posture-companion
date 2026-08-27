# AI Yoga Posture & Wellness Companion
#
# The container runs everything that does not need a camera or a window: the
# engine self-test, the simulated practice session, the reference fitting and
# the held-out validation.  That is the whole evidence base for the project, so
# a reviewer with only Docker installed can reproduce every number in the docs
# without installing Python at all.
#
# The live trainer needs a webcam and a display.  On Linux that works from the
# container (see compose.yaml, service `live`); on Windows and macOS Docker
# cannot reach the webcam, so run the trainer natively there - `python
# bootstrap.py --run` sets that up in one command.

FROM python:3.12-slim-bookworm

# OpenCV needs these even in headless use; curl fetches the pose model.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so edits to the source do not invalidate the layer.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# The pose model is baked in, so the image runs with no network at all.
RUN mkdir -p models \
    && curl -fSL -o models/pose_landmarker_full.task \
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" \
    && curl -fSL -o models/pose_landmarker_lite.task \
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

COPY . .

# Fail the build if the engine is broken - a green image means a green engine.
RUN python tools/selftest.py

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    QT_QPA_PLATFORM=offscreen

CMD ["python", "tools/selftest.py"]
