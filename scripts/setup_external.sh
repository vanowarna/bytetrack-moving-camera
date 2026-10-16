#!/usr/bin/env bash
# Linux/macOS equivalent of setup_external.ps1
set -euo pipefail
clone() { [ -d "external/$1" ] || git clone "$2" "external/$1"; git -C "external/$1" fetch -q origin; git -C "external/$1" checkout -q "$3"; echo "$1 @ $3"; }
mkdir -p external
clone ByteTrack https://github.com/ifzhang/ByteTrack.git d1bf0191adff59bc8fcfeaa0b33d3d1642552a99
clone TrackEval https://github.com/JonathonLuiten/TrackEval.git 12c8791b303e0a0b50f753af204249e622d0281a
SITE=$(uv run python -c "import site; print(site.getsitepackages()[-1])")
realpath external/TrackEval > "$SITE/trackeval_local.pth"
echo "Done. Next: uv run python scripts/check_env.py"
