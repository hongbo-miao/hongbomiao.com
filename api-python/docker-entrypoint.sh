#!/bin/sh
set -e

. /venv/bin/activate
exec uvicorn app:app --app-dir=src --host=0.0.0.0 --port=35903
