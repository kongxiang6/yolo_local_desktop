@echo off
setlocal
cd /d "%~dp0"

set "DIST_DIR=%~dp0dist"
if not exist "%DIST_DIR%" (
    echo dist folder not found. Please run build_exe.ps1 first.
    pause
    exit /b 1
)

for /f "delims=" %%I in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "$exe = Get-ChildItem -LiteralPath ''%DIST_DIR%'' -File -Recurse | Where-Object { $_.Extension -eq ''.exe'' -and $_.DirectoryName -notmatch '\\\\_internal(\\\\|$)' } | Sort-Object FullName | Select-Object -First 1 -ExpandProperty FullName; if ($exe) { Write-Output $exe }"') do set "APP_EXE=%%I"

if not defined APP_EXE (
    echo Application exe not found in dist. Please package the project first.
    pause
    exit /b 1
)

start "" "%APP_EXE%"
