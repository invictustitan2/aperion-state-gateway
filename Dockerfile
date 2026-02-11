# Aperion Cortex Container
# Unified memory and state management service

FROM python:3.11-slim

LABEL maintainer="Aperion Team <team@aperion.ai>"
LABEL description="The Cortex - Unified memory and state management service"
LABEL version="0.1.0"

# Set environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app user
RUN groupadd -r cortex && useradd -r -g cortex cortex

# Create directories
WORKDIR /app
RUN mkdir -p /app/data && chown -R cortex:cortex /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install hatch && \
    pip install . --no-deps

# Copy application code
COPY src/ ./src/

# Install the package
RUN pip install .

# Switch to non-root user
USER cortex

# Expose port
EXPOSE 4949

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:4949/healthz').raise_for_status()" || exit 1

# Default command
CMD ["aperion-cortex", "--host", "0.0.0.0", "--port", "4949"]
