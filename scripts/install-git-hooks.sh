#!/usr/bin/env bash
# Point this repo at .githooks so pre-push (Ruff) runs automatically.
set -euo pipefail
cd "$(dirname "$0")/.."
git config core.hooksPath .githooks
echo "core.hooksPath set to .githooks — pre-push will run Ruff before git push."
