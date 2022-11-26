#!/bin/sh
set -e

. /venv/bin/activate
gunicorn 'hm_api_python:create_app()' --bind=:35903 --workers=5
