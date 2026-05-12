param(
    [string]$Python = "python",
    [string]$Prompt = "A cinematic drone shot over neon cyberpunk city streets at dusk, volumetric fog, dramatic lighting",
    [string]$NegativePrompt = "Bright tones, overexposed, static, blurred details, subtitles, paintings, low quality, ugly, deformed, disfigured, still picture, messy background",
    [string]$QnnManifest = "D:\platform-tools\wan21_13b_work\qnn\wan_t2v_1p3b_832x480_17f_seq128_qnn_manifest.json",
    [string]$QnnManifestDir = "",
    [int]$Width = 0,
    [int]$Height = 0,
    [switch]$ExactResolution,
    [string]$ModelDir = "D:\platform-tools\wan21_13b_work\official_core\official-diffusers",
    [string]$TextEncoderDir = "D:\platform-tools\wan21_13b_work\int8_text_encoder\int8-diffusers",
    [string]$SdkRoot = "D:\platform-tools\sdxl_npu\qairt_2.44\qairt\2.44.0.260225",
    [string]$NdkRoot = "C:\Users\vital\AppData\Local\Android\Sdk\ndk\28.2.13676358",
    [string]$OutputDir = (Join-Path $PSScriptRoot "output\end_to_end"),
    [string]$PhoneBase = "/data/local/tmp/wan21_t2v_qnn",
    [ValidateSet("host", "phone")]
    [string]$VaeBackend = "host",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$HostVaeDevice = "auto",
    [int]$Steps = 8,
    [double]$GuidanceScale = 1.0,
    [int]$Seed = 1234,
    [int]$Fps = 16,
    [double]$MinDurationSec = 1.0,
    [int]$TargetFrames = 0,
    [double]$TargetDurationSec = 0.0,
    [ValidateSet("hold_last", "loop")]
    [string]$UpsampleFill = "hold_last",
    [string]$Adb = "",
    [string]$Serial = "",
    [switch]$SkipDeploy,
    [switch]$SkipContextGen,
    [switch]$KeepModelLibsAfterContext
)

$ErrorActionPreference = "Stop"

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

$HostScript = Join-Path $PSScriptRoot "host_phone_wan_generate.py"
if (-not (Test-Path $HostScript)) {
    throw "host_phone_wan_generate.py not found: $HostScript"
}

if (($Width -gt 0) -xor ($Height -gt 0)) {
    throw "Provide both -Width and -Height, or leave both as 0"
}

Write-Host ""
Write-Host "WAN 2.1 end-to-end wrapper (beta)"
Write-Warning "WAN E2E path НЕ ПРОВЕРЕН и возможно вовсе не работает. Используйте как тестовый workflow."
Write-Host ""
Write-Host "Prompt        : $Prompt"
Write-Host "Manifest      : $QnnManifest"
if (-not [string]::IsNullOrWhiteSpace($QnnManifestDir)) {
    Write-Host "Manifest dir  : $QnnManifestDir"
}
if ($Width -gt 0 -and $Height -gt 0) {
    Write-Host "Requested WxH : ${Width}x${Height}"
}
Write-Host "Output dir    : $OutputDir"
Write-Host "Phone base    : $PhoneBase"

$Command = @(
    $Python,
    $HostScript,
    "--prompt", $Prompt,
    "--negative-prompt", $NegativePrompt,
    "--qnn-manifest", $QnnManifest,
    "--model-dir", $ModelDir,
    "--text-encoder-dir", $TextEncoderDir,
    "--sdk-root", $SdkRoot,
    "--ndk-root", $NdkRoot,
    "--output-dir", $OutputDir,
    "--phone-base", $PhoneBase,
    "--vae-backend", $VaeBackend,
    "--host-vae-device", $HostVaeDevice,
    "--steps", $Steps.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--guidance-scale", $GuidanceScale.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--seed", $Seed.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--fps", $Fps.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--min-duration-sec", $MinDurationSec.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--target-frames", $TargetFrames.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--target-duration-sec", $TargetDurationSec.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--upsample-fill", $UpsampleFill
)

if (-not [string]::IsNullOrWhiteSpace($QnnManifestDir)) {
    $Command += @("--qnn-manifest-dir", $QnnManifestDir)
}
if ($Width -gt 0 -and $Height -gt 0) {
    $Command += @(
        "--width", $Width.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--height", $Height.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    )
}
if ($ExactResolution) {
    $Command += "--exact-resolution"
}
if (-not [string]::IsNullOrWhiteSpace($Adb)) {
    $Command += @("--adb", $Adb)
}
if (-not [string]::IsNullOrWhiteSpace($Serial)) {
    $Command += @("--serial", $Serial)
}
if ($SkipDeploy) {
    $Command += "--skip-deploy"
}
if ($SkipContextGen) {
    $Command += "--skip-context-gen"
}
if ($KeepModelLibsAfterContext) {
    $Command += "--no-delete-model-libs-after-context"
}

Invoke-Step -Title "WAN host-phone orchestrated run" -Command $Command

Write-Host ""
Write-Host ("=" * 72)
Write-Host "Done"
Write-Host ("=" * 72)
Write-Host "Output root: $OutputDir"
Write-Host ""
Write-Host "Notes:"
Write-Host "- This script calls host_phone_wan_generate.py and keeps WAN in explicit beta mode."
Write-Host "- Use -Width/-Height to request hot bucket selection across available manifests."
Write-Host "- Add -ExactResolution to fail instead of snapping to nearest available bucket."