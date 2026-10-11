#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Load local configuration if the user copied .env.example to .env.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
exec python3 auto_trainer_v3.py
