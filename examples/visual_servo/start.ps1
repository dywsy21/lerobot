param(
    [ValidateSet('simulation', 'camera', 'broker')][string]$Source = 'simulation',
    [int]$Camera = 2,
    [int]$Port = 8767,
    [switch]$AllowHardware
)
$ErrorActionPreference = 'Stop'
$courseRepository = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$courseArguments = @('run', '--no-sync', 'python', '-X', 'utf8', (Join-Path $PSScriptRoot 'experiment.py'),
                     'web', '--source', $Source, '--camera', $Camera, '--port', $Port)
if ($AllowHardware) { $courseArguments += '--allow-hardware' }
Push-Location -LiteralPath $courseRepository
try {
    & uv @courseArguments
    if ($LASTEXITCODE -ne 0) { throw "Experiment exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
