#!/bin/bash
set -e

# Install doc-classifier if mounted (copy first since volume is read-only)
if [ -d "/app/doc-classifier" ]; then
    echo "Installing doc-classifier from mounted volume..."
    cp -r /app/doc-classifier /tmp/doc-classifier
    pip install /tmp/doc-classifier --quiet
    rm -rf /tmp/doc-classifier
fi

# Run the worker
exec python worker_docid.py
