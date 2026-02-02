#!/bin/bash

# Cleanup function to kill background processes on container exit
cleanup() {
    echo "Container stopping... Killing subprocesses."
    kill $(jobs -p)
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "Starting App A: Ingestion Engine on Port 8000..."
uvicorn server:app --host 0.0.0.0 --port 8000 &

echo "Starting App B: Chat Engine on Port 8001..."
uvicorn chat_server:app --host 0.0.0.0 --port 8001 &

# Wait for any process to exit
wait -n
exit $?