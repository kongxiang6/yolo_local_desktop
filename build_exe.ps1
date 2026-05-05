param(
    [switch]$Full,
    [switch]$IncludeRuntime,
    [string]$RuntimeSource,
    [string]$SpecPath = '.\yolo_local_desktop.spec'
)

$ErrorActionPreference = 'Stop'

$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$buildDir = Join-Path $project 'build'
$distRootParent = Join-Path $project 'dist'
$releaseRoot = Join-Path $project 'release'
$deliveryDir = Join-Path $project 'delivery'
$runtimeCacheFile = Join-Path $project '.runtime_source_path.txt'
$distRoot = $null
$appName = $null
$releaseDir = $null
$resolvedSpecPath = [System.IO.Path]::GetFullPath((Join-Path $project $SpecPath))
$specBaseName = [System.IO.Path]::GetFileNameWithoutExtension($resolvedSpecPath)
$releaseZipPath = Join-Path $releaseRoot "${specBaseName}_release.zip"

if (-not (Test-Path -LiteralPath $resolvedSpecPath)) {
    throw "Spec file was not found: $resolvedSpecPath"
}

function Test-RuntimeDirectory {
    param([string]$PathValue)
    if (-not $PathValue) {
        return $false
    }
    return (Test-Path -LiteralPath (Join-Path $PathValue 'python\python.exe'))
}

function Resolve-RuntimeSource {
    param([string]$PreferredPath)

    $candidates = @()
    if ($PreferredPath) {
        $candidates += $PreferredPath
    }
    if ($env:YOLO_TOOL_RUNTIME_SOURCE) {
        $candidates += $env:YOLO_TOOL_RUNTIME_SOURCE
    }
    if (Test-Path -LiteralPath $runtimeCacheFile) {
        $cachedPath = (Get-Content -LiteralPath $runtimeCacheFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
        if ($cachedPath) {
            $candidates += $cachedPath
        }
    }
    if ($releaseDir) {
        $candidates += (Join-Path $releaseDir 'runtime')
    }
    $candidates += (Join-Path $project 'runtime')

    foreach ($candidate in $candidates) {
        if (Test-RuntimeDirectory $candidate) {
            Set-Content -LiteralPath $runtimeCacheFile -Value $candidate -Encoding UTF8
            return $candidate
        }
    }
    return $null
}

function Ensure-ParentDirectory {
    param([string]$TargetPath)
    $parent = Split-Path -Parent $TargetPath
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
}

Set-Location $project

$runtimeSourceResolved = $null
if ($IncludeRuntime) {
    $runtimeSourceResolved = Resolve-RuntimeSource -PreferredPath $RuntimeSource
    if (-not $runtimeSourceResolved) {
        throw 'Bundled runtime source was not found. Use -RuntimeSource to specify a valid runtime directory.'
    }
}

$pyInstallerArgs = @($resolvedSpecPath, '--noconfirm')
if ($Full) {
    $pyInstallerArgs += '--clean'
    $env:YOLO_TOOL_BUILD_FAST = '0'
}
else {
    $env:YOLO_TOOL_BUILD_FAST = '1'
}

python -m PyInstaller @pyInstallerArgs

$distRoot = Get-ChildItem -LiteralPath $distRootParent -Directory -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $distRoot -or -not (Test-Path -LiteralPath $distRoot)) {
    throw 'PyInstaller did not produce the expected dist directory.'
}
$appName = Split-Path -Leaf $distRoot
$releaseDir = Join-Path $releaseRoot $appName

New-Item -ItemType Directory -Force -Path (Join-Path $distRoot 'presets\train') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $distRoot 'presets\val') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $distRoot 'presets\predict') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $distRoot 'presets\track') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $distRoot 'presets\export') | Out-Null

$distRuntime = Join-Path $distRoot 'runtime'
if (Test-Path -LiteralPath $distRuntime) {
    Remove-Item -LiteralPath $distRuntime -Recurse -Force
}
if ($IncludeRuntime) {
    if ($Full) {
        Copy-Item -LiteralPath $runtimeSourceResolved -Destination $distRuntime -Recurse -Force
    }
    else {
        New-Item -ItemType Junction -Path $distRuntime -Target $runtimeSourceResolved | Out-Null
    }
}
elseif (-not $Full) {
    $projectRuntime = Join-Path $project 'runtime'
    if (Test-RuntimeDirectory $projectRuntime) {
        $runtimeSourceResolved = $projectRuntime
    }
    elseif ($RuntimeSource) {
        $runtimeSourceResolved = Resolve-RuntimeSource -PreferredPath $RuntimeSource
    }
    if ($runtimeSourceResolved) {
        if (Test-Path -LiteralPath $distRuntime) {
            Remove-Item -LiteralPath $distRuntime -Recurse -Force
        }
        New-Item -ItemType Junction -Path $distRuntime -Target $runtimeSourceResolved | Out-Null
    }
}
if ($IncludeRuntime -and -not (Test-Path -LiteralPath (Join-Path $distRuntime 'python\python.exe'))) {
    throw 'Bundled runtime setup failed.'
}

if (Test-Path -LiteralPath (Join-Path $deliveryDir 'README_FOR_SHARE.txt')) {
    Copy-Item -LiteralPath (Join-Path $deliveryDir 'README_FOR_SHARE.txt') -Destination (Join-Path $distRoot 'README_FOR_SHARE.txt') -Force
}
$localGuide = Join-Path $project 'USER_GUIDE.txt'
if (Test-Path -LiteralPath $localGuide) {
    Copy-Item -LiteralPath $localGuide -Destination (Join-Path $distRoot (Split-Path $localGuide -Leaf)) -Force
}
$licenseFile = Join-Path $project 'LICENSE'
if (Test-Path -LiteralPath $licenseFile) {
    Copy-Item -LiteralPath $licenseFile -Destination (Join-Path $distRoot 'LICENSE') -Force
}

if ($Full) {
    New-Item -ItemType Directory -Force -Path $releaseRoot | Out-Null
    if (Test-Path -LiteralPath $releaseDir) {
        Remove-Item -LiteralPath $releaseDir -Recurse -Force
    }
    Copy-Item -LiteralPath $distRoot -Destination $releaseDir -Recurse -Force

    $buildExe = Get-ChildItem -LiteralPath (Join-Path $buildDir $specBaseName) -Filter '*.exe' -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
    if ($buildExe) {
        Remove-Item -LiteralPath $buildExe -Force
    }

    if (Test-Path -LiteralPath $releaseZipPath) {
        Remove-Item -LiteralPath $releaseZipPath -Force
    }
    Compress-Archive -Path $releaseDir -DestinationPath $releaseZipPath
}

Write-Host "Build mode: $(if ($Full) { 'full release' } else { 'fast local' })"
Write-Host "Spec file: $resolvedSpecPath"
Write-Host "Include runtime: $IncludeRuntime"
if ($runtimeSourceResolved) {
    Write-Host "Runtime source: $runtimeSourceResolved"
}
else {
    Write-Host "Runtime source: <not bundled>"
}
Write-Host "Dist ready: $distRoot"
if ($Full) {
    Write-Host "Release ready: $releaseDir"
    Write-Host "Zip ready: $releaseZipPath"
}
else {
    Write-Host "Fast build skips release copy and zip compression. Use -Full when you need a share/release package."
}
