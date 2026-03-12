FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install build dependencies to compile extensions (like tiktoken/tokenizers)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the sentence-transformers model during build phase to cache it in the image file
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the application
COPY . .

# Expose Gradio port
EXPOSE 7860

# We need the user to either run Ollama inside the container (complex) 
# or point to it via OLLAMA_BASE_URL (standard approach).
# Default to host.docker.internal for local docker testing
ENV OLLAMA_BASE_URL="http://host.docker.internal:11434"
ENV PORT=7860

# Command to run the application
CMD ["python", "app.py"]
