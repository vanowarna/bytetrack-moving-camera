# Clones pinned third-party code into external/ (not committed).
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File .\scripts\setup_external.ps1
# Note: uses -LiteralPath everywhere because the repo path may contain [ ] characters.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

$repos = @(
  @{ name = "ByteTrack"; url = "https://github.com/ifzhang/ByteTrack.git";        commit = "d1bf0191adff59bc8fcfeaa0b33d3d1642552a99" },
  @{ name = "TrackEval"; url = "https://github.com/JonathonLuiten/TrackEval.git"; commit = "12c8791b303e0a0b50f753af204249e622d0281a" }
)
$ext = Join-Path $root "external"
if (-not (Test-Path -LiteralPath $ext)) { New-Item -ItemType Directory -Path $ext | Out-Null }
foreach ($r in $repos) {
  $dst = Join-Path $ext $r.name
  if (-not (Test-Path -LiteralPath (Join-Path $dst ".git"))) { git clone $r.url $dst }
  git -C $dst fetch --quiet origin
  git -C $dst checkout --quiet $r.commit
  Write-Host "$($r.name) @ $($r.commit)"
}
# Expose TrackEval as an importable package via a .pth file in the uv venv
$site = (uv run python -c "import site; print(site.getsitepackages()[-1])").Trim()
$pth = Join-Path $site "trackeval_local.pth"
Set-Content -LiteralPath $pth -Value (Join-Path $ext "TrackEval")
Write-Host "Wrote $pth"
Write-Host "Done. Next: uv run python scripts/check_env.py"
