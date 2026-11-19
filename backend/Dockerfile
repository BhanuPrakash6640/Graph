FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU and PyG CPU wheels
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu \
    torchvision==0.20.1+cpu \
    torchaudio==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    torch-geometric==2.6.1

# Install backend dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install ML dependencies needed for inference
COPY pyproject.toml ./
COPY ml/ ./ml/
RUN pip install --no-cache-dir -e .

# Copy backend application code
COPY backend/ ./backend/

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

