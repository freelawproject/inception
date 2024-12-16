#!/bin/sh
set -e

if [ "$TARGET_ENV" = "prod" ]; then
    # Production command
    exec gunicorn \
        --workers=${EMBEDDING_WORKERS:-4} \
        --worker-class=uvicorn.workers.UvicornWorker \
        --bind=0.0.0.0:8005 \
        --timeout=300 \
        --access-logfile=- \
        --error-logfile=- \
        inception.main:app
else
    # Development command
    exec uvicorn inception.main:app \
        --host 0.0.0.0 \
        --port 8005 \
        --log-level debug \
        --reload
fi