FROM python:3.12-slim

WORKDIR /app

# System deps (optional but often useful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . /app

# Expose port
EXPOSE 8000

# Start API
CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]

