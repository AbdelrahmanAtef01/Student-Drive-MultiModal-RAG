# -----------------------------------------------------------------------------
# STAGE 1: Base Image (CUDA Runtime, Ubuntu)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# -----------------------------------------------------------------------------
# STAGE 2: System Dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    ffmpeg \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# -----------------------------------------------------------------------------
# STAGE 3: Python Dependencies
# -----------------------------------------------------------------------------
WORKDIR /app

COPY requirements.txt .

# Upgrade pip and install deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# STAGE 4: Application Code
# -----------------------------------------------------------------------------
COPY . .

# Ensure the startup script is executable
RUN chmod +x start_servers.sh

# Expose both ports
EXPOSE 8000
EXPOSE 8001

# -----------------------------------------------------------------------------
# STAGE 5: Runtime
# -----------------------------------------------------------------------------
CMD ["./start_servers.sh"]