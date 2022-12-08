#!/bin/sh
set -e

. /venv/bin/activate
gunicorn 'flaskr:create_app()' --bind=:35903 --workers=5
