FROM python:3.14.5-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/xrecorder
COPY requirements.lock ./requirements.lock
RUN python -m pip install --no-cache-dir -r requirements.lock

CMD ["bash"]
