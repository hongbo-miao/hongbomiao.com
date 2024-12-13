#!/bin/sh
set -e

. /venv/bin/activate
exec uvicorn main:app --host=0.0.0.0 --port=35903
