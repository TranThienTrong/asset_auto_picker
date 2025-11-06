FROM python:3.11-slim-bullseye AS release

ENV WORKSPACE_ROOT=/app/
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    libglib2.0-dev \
    libnss3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
RUN pip install --no-cache-dir uv

WORKDIR $WORKSPACE_ROOT

# Copy requirements file
COPY requirements.txt $WORKSPACE_ROOT

# Install dependencies using uv
RUN uv pip install --system -r requirements.txt

# Copy the rest of the code
COPY . $WORKSPACE_ROOT

# Expose port 8000
EXPOSE 8000

# No CMD - Railway will use startCommand from railway.toml
CMD ["run", "uvicorn", "emoji_agent:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]