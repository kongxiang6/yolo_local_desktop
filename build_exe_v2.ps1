param(
    [switch]$Full,
    [switch]$IncludeRuntime,
    [string]$RuntimeSource
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $scriptDir 'build_exe.ps1') -Full:$Full -IncludeRuntime:$IncludeRuntime -RuntimeSource $RuntimeSource -SpecPath '.\yolo_local_desktop_v2.spec'
