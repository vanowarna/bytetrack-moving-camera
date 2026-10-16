# Clones pinned third-party code into external/ (not committed).
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File .\scripts\setup_external.ps1
$ErrorActionPreference = "Stop"
$repos = @(
  @{ name = "ByteTrack"; url = "https://github.com/ifzhang/ByteTrack.git";        commit = "d1bf0191adff59bc8fcfeaa0b33d3d1642552a99" },
  @{ name = "TrackEval"; url = "https://github.com/JonathonLuiten/TrackEval.git"; commit = "12c8791b303e0a0b50f753af204249e622d0281a" }
)
New-Item -ItemType Directory -Force -Path "external" | Out-Null
foreach ($r in $repos) {
  $dst = Join-Path "external" $r.name
  if (-not (Test-Path -LiteralPath $dst)) { git clone $r.url $dst }
  git -C $dst fetch --quiet origin
  git -C $dst checkout --quiet $r.commit
  Write-Host "$($r.name) @ $($r.commit)"
}
# TrackEval is imported as a package; expose it on the path via a .pth file in the uv venv
$site = (uv run python -c "import site; print(site.getsitepackages()[0])").Trim()
Set-Content -LiteralPath (Join-Path $site "trackeval_local.pth") -Value (Resolve-Path "external/TrackEval").Path
Write-Host "Done. Next: uv run python scripts/check_env.py"
