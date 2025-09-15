# Multi-stage Docker build for K12 Math Education Digital Ecosystem
FROM node:18-alpine AS frontend-builder

# Build homework system frontend
WORKDIR /app/homework_system
COPY homework_system/package*.json ./
RUN npm ci --only=production
COPY homework_system/ ./
RUN npm run build

# Build symbol keyboard frontend
WORKDIR /app/symbol-frontend
COPY Subject_symbol_dynamic_keyboard/board-frontend/package*.json ./
RUN npm ci --only=production
COPY Subject_symbol_dynamic_keyboard/board-frontend/ ./
RUN npm run build

# Python backend stage
FROM python:3.9-slim AS backend

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python requirements and install dependencies
COPY homework-backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY homework-backend/ ./

# Copy built frontend files
COPY --from=frontend-builder /app/homework_system/dist ./static/homework
COPY --from=frontend-builder /app/symbol-frontend/dist ./static/symbol

# Create necessary directories
RUN mkdir -p logs data/uploads

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Start command
CMD ["python", "app.py"]
