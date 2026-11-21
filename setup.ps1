# VidForge setup: creates a Python environment, installs VidForge, and downloads FFmpeg (and
# NVEncC on NVIDIA PCs). Everything stays inside this folder. Safe to run again.
param(
    [switch]$SkipTools  # don't download FFmpeg / NVEncC now (they can be installed from the System page)
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Find-Python312 {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $exe = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
    }
    $guess = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
    if (Test-Path -LiteralPath $guess) { return $guess }
    return $null
}

Write-Host "== VidForge setup ==" -ForegroundColor Cyan
$python = Find-Python312
if (-not $python) {
    Write-Host "Python 3.12 is required but was not found."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        $answer = Read-Host "Install Python 3.12 now with winget? (y/n)"
        if ($answer -match '^[yY]') {
            winget install --id Python.Python.3.12 -e --scope user --accept-package-agreements
            $python = Find-Python312
        }
    }
    if (-not $python) {
        Write-Host "Please install Python 3.12 from https://www.python.org/downloads/ and run this again." -ForegroundColor Yellow
        exit 1
    }
}
Write-Host "Using Python: $python"

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    Write-Host "Creating the Python environment (.venv)..."
    & $python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "could not create .venv" }
}
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Host "Installing VidForge and its packages (first time: a few minutes, ~1 GB)..."
& $venvPy -m pip install --upgrade pip --quiet
& $venvPy -m pip install -e . --quiet
if ($LASTEXITCODE -ne 0) { throw "package installation failed" }

# Keep tools and models inside this folder (portable mode).
New-Item -ItemType Directory -Force -Path "vidforge-data" | Out-Null

if (-not $SkipTools) {
    $vf = Join-Path $PSScriptRoot ".venv\Scripts\vidforge.exe"
    Write-Host "Downloading FFmpeg (~200 MB, checksum-verified)..."
    & $vf tools fetch ffmpeg
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        Write-Host "NVIDIA GPU found: downloading NVEncC for Fast mode (~285 MB)..."
        & $vf tools fetch nvencc
    }
    Write-Host ""
    & $vf info
}

Write-Host ""
Write-Host "Setup finished. Start VidForge with 'Start VidForge.bat'." -ForegroundColor Green
