#!/bin/bash
# run_docker.sh
# Helper script to build and run the RAG system in Docker.

echo "🐳 Building RAG System Docker Image..."
docker build -t rag_system:latest .

echo "🚀 Starting RAG System Container..."
echo "Note: Ensure Ollama is running locally and has the 'llama3.2:3b' model installed!"
echo "If Ollama is on MacOS/Windows, host.docker.internal will route to it automatically."
echo "If on Linux, you may need to add: --add-host=host.docker.internal:host-gateway"

docker run -d \
  --name rag_app \
  -p 7860:7860 \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/.env:/app/.env \
  --add-host=host.docker.internal:host-gateway \
  rag_system:latest

echo "✅ Container started!"
echo "📍 Access the app at: http://localhost:7860"
echo "📜 View logs with: docker logs -f rag_app"
