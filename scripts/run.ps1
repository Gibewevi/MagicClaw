param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$MagicArgs
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$LogDir = Join-Path $Root ".magicclaw\logs"
$BootstrapLog = Join-Path $LogDir "bootstrap.log"
$InstallMarker = Join-Path $Venv ".magicclaw-install.hash"
$Pyproject = Join-Path $Root "pyproject.toml"

Set-Location $Root

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

function ConvertTo-ProcessArgument {
    param([AllowNull()][string]$Value)

    if ($null -eq $Value -or $Value.Length -eq 0) {
        return '""'
    }
    if ($Value -notmatch '[\s"]') {
        return $Value
    }

    $escaped = $Value -replace '(\\*)"', '$1$1\"'
    $escaped = $escaped -replace '(\\+)$', '$1$1'
    return '"' + $escaped + '"'
}

function Invoke-SpinnerProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message,
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $BootstrapLog -Value ""
    Add-Content -Path $BootstrapLog -Value "[$timestamp] $Message"
    Add-Content -Path $BootstrapLog -Value "> $FilePath $($Arguments -join ' ')"

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $FilePath
    $psi.Arguments = ($Arguments | ForEach-Object { ConvertTo-ProcessArgument $_ }) -join " "
    $psi.WorkingDirectory = $Root
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $psi
    [void]$process.Start()

    $frames = @("|", "/", "-", "\")
    $index = 0
    while (-not $process.HasExited) {
        $frame = $frames[$index % $frames.Count]
        Write-Host "`r$frame $Message" -NoNewline
        Start-Sleep -Milliseconds 120
        $index++
        $process.Refresh()
    }
    $outText = $process.StandardOutput.ReadToEnd()
    $errText = $process.StandardError.ReadToEnd()
    $process.WaitForExit()
    if ($outText) { Add-Content -Path $BootstrapLog -Value $outText }
    if ($errText) { Add-Content -Path $BootstrapLog -Value $errText }

    Write-Host "`r$(' ' * 100)`r" -NoNewline
    $exitCode = if ($null -eq $process.ExitCode) { 1 } else { $process.ExitCode }
    if ($exitCode -ne 0) {
        Write-Host "[ERROR] $Message"
        Write-Host "Consulte les logs: $BootstrapLog"
        if ($errText) {
            Write-Host ""
            Write-Host ($errText.Trim().Split([Environment]::NewLine) | Select-Object -Last 8) -Separator [Environment]::NewLine
        }
        exit $exitCode
    }

    Write-Host "[OK] $Message"
}

if (-not (Test-Path $Python)) {
    Invoke-SpinnerProcess "Creation de l'environnement Python" "python" @("-m", "venv", $Venv)
}

$projectHash = (Get-FileHash -Path $Pyproject -Algorithm SHA256).Hash
$installedHash = if (Test-Path $InstallMarker) { (Get-Content -Raw $InstallMarker).Trim() } else { "" }
$dependenciesReady = $false

if ($installedHash -eq $projectHash) {
    & $Python -c "import magic_claw, rich, pydantic, psutil, httpx, huggingface_hub, dotenv" *> $null
    $dependenciesReady = ($LASTEXITCODE -eq 0)
}

if (-not $dependenciesReady) {
    Invoke-SpinnerProcess "Preparation de pip" $Python @("-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--no-input", "-U", "pip")
    Invoke-SpinnerProcess "Installation des dependances Magic Claw" $Python @("-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--no-input", "-e", ".")
    Set-Content -Path $InstallMarker -Value $projectHash
}

if (-not $MagicArgs -or $MagicArgs.Count -eq 0) {
    & $Python -m magic_claw
} else {
    & $Python -m magic_claw @MagicArgs
}
