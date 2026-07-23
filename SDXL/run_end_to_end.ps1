param(
    [string]$Checkpoint,

    [string]$DefaultCheckpoint = "J:\ComfyUI\models\checkpoints\waiIllustriousSDXL_v170.safetensors",

    [string]$Python = "python",
    [string]$OutputRoot = (Join-Path (Split-Path $PSScriptRoot -Parent) "build\sdxl_work"),
    [string]$Resolution = "1024x1024",
    [string]$HotSwapResolutions = "",
    [string]$LightningLora = "",
    [double]$LightningLoraScale = 1.0,
    [string]$HotSwapLoras = "",
    [string]$HotSwapLoraScales = "",
    [switch]$VerifyMerge,
    [string]$ContextsDir,
    [string]$QnnLibDir,
    [string]$QnnBinDir,
    [string]$PhoneBase = "/sdcard/Download/sdxl_qnn",
    [string]$Prompt,
    [int]$Seed = 42,
    [switch]$SkipBuild,
    [switch]$SkipDeploy,
    [switch]$SkipSmokeTest
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path $PSScriptRoot -Parent
$ScriptsDir = Join-Path $RepoRoot "scripts"
$BuildHelper = Join-Path $ScriptsDir "build_all.py"
$DeployHelper = Join-Path $ScriptsDir "deploy_to_phone.py"
$AdbLocal = Join-Path $RepoRoot "adb.exe"
$AdbExternal = "D:\platform-tools\adb.exe"
$Adb = if (Test-Path $AdbLocal) {
    $AdbLocal
}
elseif (Test-Path $AdbExternal) {
    $AdbExternal
}
else {
    "adb"
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Title,

        [Parameter(Mandatory = $true)]
        [string[]]$Command
    )

    Write-Host ""
    Write-Host ("=" * 72)
    Write-Host "[STEP] $Title"
    Write-Host ("=" * 72)
    Write-Host ("  " + ($Command -join " "))

    $cmdExe = $Command[0]
    $cmdArgs = @()
    if ($Command.Length -gt 1) {
        $cmdArgs = $Command[1..($Command.Length - 1)]
    }

    & $cmdExe @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Title (exit code $LASTEXITCODE)"
    }
}

Write-Host ""
Write-Host "Model-to-NPU SDXL beta flow"
Write-Host "Repo root : $RepoRoot"

if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
    $enteredCheckpoint = Read-Host "Path to SDXL checkpoint (.safetensors) [$DefaultCheckpoint]"
    if ([string]::IsNullOrWhiteSpace($enteredCheckpoint)) {
        $Checkpoint = $DefaultCheckpoint
    }
    else {
        $Checkpoint = $enteredCheckpoint.Trim()
    }
}

$Checkpoint = $Checkpoint.Trim('"')

Write-Host "Checkpoint: $Checkpoint"
Write-Host "Output    : $OutputRoot"
Write-Host "Resolution: $Resolution"
if (-not [string]::IsNullOrWhiteSpace($HotSwapResolutions)) {
    Write-Host "Hot-swap  : $HotSwapResolutions"
}
if (-not [string]::IsNullOrWhiteSpace($LightningLora)) {
    Write-Host "LoRA base : $LightningLora (scale=$LightningLoraScale)"
}
if (-not [string]::IsNullOrWhiteSpace($HotSwapLoras)) {
    Write-Host "LoRA slots: $HotSwapLoras"
}
Write-Host "Phone base: $PhoneBase"

if (-not (Test-Path $Checkpoint)) {
    throw "Checkpoint not found: $Checkpoint"
}

if (-not $SkipBuild) {
    $BuildArgs = @(
        $Python,
        $BuildHelper,
        "--checkpoint", $Checkpoint,
        "--output-dir", $OutputRoot,
        "--resolution", $Resolution
    )

    if (-not [string]::IsNullOrWhiteSpace($HotSwapResolutions)) {
        $BuildArgs += @("--hot-swap-resolutions", $HotSwapResolutions)
    }
    if (-not [string]::IsNullOrWhiteSpace($LightningLora)) {
        $BuildArgs += @("--lightning-lora", $LightningLora)
        $BuildArgs += @("--lightning-lora-scale", $LightningLoraScale.ToString([System.Globalization.CultureInfo]::InvariantCulture))
    }
    if (-not [string]::IsNullOrWhiteSpace($HotSwapLoras)) {
        $BuildArgs += @("--hot-swap-loras", $HotSwapLoras)
    }
    if (-not [string]::IsNullOrWhiteSpace($HotSwapLoraScales)) {
        $BuildArgs += @("--hot-swap-lora-scales", $HotSwapLoraScales)
    }
    if ($VerifyMerge) {
        $BuildArgs += @("--verify-merge")
    }

    Invoke-Step -Title "Early reproducible SDXL build stages" -Command $BuildArgs
}
else {
    Write-Host "[skip] Build stage skipped."
}

if (-not $SkipDeploy) {
    if (-not $ContextsDir) {
        Write-Warning "Deployment skipped because -ContextsDir was not provided. The current public beta runtime still expects already-built split context binaries (CLIP/CLIP-G/VAE/unet_encoder/unet_decoder)."
    }
    else {
        $DeployArgs = @(
            $Python,
            $DeployHelper,
            "--adb", $Adb,
            "--contexts-dir", $ContextsDir,
            "--phone-base", $PhoneBase
        )

        if ($QnnLibDir) {
            $DeployArgs += @("--qnn-lib-dir", $QnnLibDir)
        }
        if ($QnnBinDir) {
            $DeployArgs += @("--qnn-bin-dir", $QnnBinDir)
        }

        Invoke-Step -Title "Deploy runtime files to phone" -Command $DeployArgs
    }
}
else {
    Write-Host "[skip] Deploy stage skipped."
}

if (-not $SkipSmokeTest -and $Prompt) {
    $SmokeCommand = "export PATH=/data/data/com.termux/files/usr/bin:`$PATH && export SDXL_QNN_BASE=$PhoneBase && python3 $PhoneBase/phone_gen/generate.py `"$Prompt`" --seed $Seed"
    Invoke-Step -Title "Phone-side smoke generation" -Command @(
        $Adb,
        "shell",
        $SmokeCommand
    )
}
elseif (-not $SkipSmokeTest) {
    Write-Host "[info] Smoke test not run because -Prompt was not provided."
}
else {
    Write-Host "[skip] Smoke test skipped."
}

Write-Host ""
Write-Host ("=" * 72)
Write-Host "Done"
Write-Host ("=" * 72)
Write-Host "Artifacts (early reproducible stages): $OutputRoot"
Write-Host ""
Write-Host "Notes:"
Write-Host "- If -Checkpoint is omitted, the script asks interactively and defaults to $DefaultCheckpoint."
Write-Host "- This script follows the current public beta path of the repo."
Write-Host "- The default build is 1024x1024; pass -Resolution and optional -HotSwapResolutions for extra buckets."
Write-Host "- HotSwap LoRA supports up to 4 pre-baked slots via -HotSwapLoras (comma-separated)."
Write-Host "- The build step covers checkpoint -> diffusers -> Lightning merge -> QNN-friendly UNet/CLIP/VAE ONNX export."
Write-Host "- The deploy step assumes split context binaries already exist, because that remains the current documented runtime path."
Write-Host "- The deeper Lightning/QNN lab scripts are documented in SDXL/SCRIPTS_OVERVIEW*.md and are still marked experimental."
