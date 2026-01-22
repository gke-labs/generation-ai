FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install dependencies
# accelerate is needed for device_map="auto"
RUN pip install --no-cache-dir transformers accelerate

# Copy source code
COPY src/ /app/src/

ENTRYPOINT ["python", "src/main.py"]
