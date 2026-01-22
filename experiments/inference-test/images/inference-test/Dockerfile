FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

WORKDIR /app

# Install dependencies
# accelerate is needed for device_map="auto"
RUN pip install --no-cache-dir --break-system-packages transformers accelerate

# Copy source code
COPY src/ /app/src/

ENTRYPOINT ["python", "src/main.py"]
