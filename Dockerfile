FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV QT_X11_NO_MITSHM=1
ENV XDG_RUNTIME_DIR=/tmp/runtime-root

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb-cursor0 \
    libxcb-glx0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxext6 \
    libxkbcommon-x11-0 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY configs ./configs
COPY src ./src
COPY README.md .

RUN mkdir -p \
    /app/data \
    /app/models \
    /app/outputs \
    /tmp/matplotlib \
    /tmp/runtime-root \
    && chmod 700 /tmp/runtime-root

CMD ["python", "-m", "src.gui.main"]
