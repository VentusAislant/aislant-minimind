#!/usr/bin/env bash

set -euo pipefail

# 无论从哪个目录调用，都统一在项目根目录启动。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

streamlit run aislant_minimind/inference/web/web_demo.py "$@"
