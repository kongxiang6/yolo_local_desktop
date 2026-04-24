$ErrorActionPreference = 'Stop'

$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$buildDir = Join-Path $project 'build'
$releaseRoot = Join-Path $project 'release'
$deliveryDir = Join-Path $project 'delivery'
$githubStageRoot = Join-Path $releaseRoot 'github_release_tmp'
$runtimeSource = Get-ChildItem -LiteralPath 'I:\rj\QQ' -Directory -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq 'runtime' -and (Test-Path -LiteralPath (Join-Path $_.FullName 'python\python.exe')) } |
    Select-Object -First 1 -ExpandProperty FullName
Set-Location $project

python -m PyInstaller .\yolo_local_desktop.spec --noconfirm --clean

$distRoot = Get-ChildItem -LiteralPath (Join-Path $project 'dist') -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $distRoot) {
    throw 'PyInstaller did not produce a dist directory.'
}
New-Item -ItemType Directory -Force -Path (Join-Path $distRoot 'presets\train') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $distRoot 'presets\val') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $distRoot 'presets\predict') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $distRoot 'presets\track') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $distRoot 'presets\export') | Out-Null
if ($runtimeSource -and (Test-Path -LiteralPath $runtimeSource)) {
    Copy-Item -LiteralPath $runtimeSource -Destination (Join-Path $distRoot 'runtime') -Recurse -Force
}
if (-not (Test-Path -LiteralPath (Join-Path $distRoot 'runtime\python\python.exe'))) {
    throw 'Bundled runtime copy failed. Please ensure the original runtime folder is available.'
}
if (Test-Path -LiteralPath (Join-Path $deliveryDir 'README_FOR_SHARE.txt')) {
    Copy-Item -LiteralPath (Join-Path $deliveryDir 'README_FOR_SHARE.txt') -Destination (Join-Path $distRoot 'README_FOR_SHARE.txt') -Force
}
$localGuide = Get-ChildItem -LiteralPath $project -File -Filter '*.txt' -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -ne 'requirements.txt' } |
    Select-Object -First 1 -ExpandProperty FullName
if ($localGuide -and (Test-Path -LiteralPath $localGuide)) {
    Copy-Item -LiteralPath $localGuide -Destination (Join-Path $distRoot (Split-Path $localGuide -Leaf)) -Force
}

New-Item -ItemType Directory -Force -Path $releaseRoot | Out-Null
$releaseDir = Join-Path $releaseRoot (Split-Path $distRoot -Leaf)
if (Test-Path -LiteralPath $releaseDir) {
    Remove-Item -LiteralPath $releaseDir -Recurse -Force
}
Copy-Item -LiteralPath $distRoot -Destination $releaseDir -Recurse -Force

$buildExe = Get-ChildItem -LiteralPath (Join-Path $buildDir 'yolo_local_desktop') -Filter '*.exe' -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName
if ($buildExe) {
    Remove-Item -LiteralPath $buildExe -Force
}

$zipPath = Join-Path $releaseRoot ((Split-Path $releaseDir -Leaf) + '_share.zip')
if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -Path $releaseDir -DestinationPath $zipPath

$githubZipPath = Join-Path $releaseRoot ((Split-Path $releaseDir -Leaf) + '_github_no_runtime.zip')
$githubStage = Join-Path $githubStageRoot (Split-Path $releaseDir -Leaf)
if (Test-Path -LiteralPath $githubStageRoot) {
    Remove-Item -LiteralPath $githubStageRoot -Recurse -Force
}
if (Test-Path -LiteralPath $githubZipPath) {
    Remove-Item -LiteralPath $githubZipPath -Force
}
New-Item -ItemType Directory -Force -Path $githubStageRoot | Out-Null
Copy-Item -LiteralPath $releaseDir -Destination $githubStage -Recurse -Force
if (Test-Path -LiteralPath (Join-Path $githubStage 'runtime')) {
    Remove-Item -LiteralPath (Join-Path $githubStage 'runtime') -Recurse -Force
}
Compress-Archive -Path $githubStage -DestinationPath $githubZipPath
if (Test-Path -LiteralPath $githubStageRoot) {
    Remove-Item -LiteralPath $githubStageRoot -Recurse -Force
}

Write-Host "Build finished: $distRoot"
Write-Host "Release ready: $releaseDir"
Write-Host "Zip ready: $zipPath"
Write-Host "GitHub zip ready: $githubZipPath"
