FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update
RUN apt-get install -y libglib2.0-0 libgl1 libxcb1 curl
RUN rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .

# Install CPU-only PyTorch
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

# Download actual model from GitHub Release
RUN curl -L "https://github.com/vishl9455/FaceMask_Detector/releases/download/v1.0.0/model.keras" -o model.keras

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]