#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_dir="$project_dir/outputs"
mkdir -p -- "$output_dir"
log_file="$output_dir/demo_$(date +%Y-%m-%d_%H-%M-%S)_$$.log"

printf 'Saving output to %s\n' "$log_file"

(
    cd -- "$project_dir/mediator_demo"

    if [[ ! -x .venv/bin/python ]]; then
        python3.12 -m venv .venv
    fi

    if ! .venv/bin/python -c 'import autogen_core, autogen_ext.models.openai' 2>/dev/null; then
        .venv/bin/python -m pip install -r requirements.txt
    fi

    exec .venv/bin/python -u demo.py "$@"
) 2>&1 | (
    umask 077
    tee "$log_file"
)
