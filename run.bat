@echo off
setlocal

set IMAGE_NAME=dataprocess
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

set INPUT_DIR=%SCRIPT_DIR%\input
set OUTPUT_DIR=%SCRIPT_DIR%\output
set WEIGHTS_DIR=%SCRIPT_DIR%\src\models\weights
set FORCE_REBUILD=0

:: Check for --rebuild flag
if "%1"=="--rebuild" set FORCE_REBUILD=1
if "%1"=="-r" set FORCE_REBUILD=1

:: Create directories if they don't exist
if not exist "%INPUT_DIR%" mkdir "%INPUT_DIR%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%WEIGHTS_DIR%" mkdir "%WEIGHTS_DIR%"



:: Force rebuild if requested
if %FORCE_REBUILD%==1 (
    echo Force rebuild requested - removing old image...
    docker rmi %IMAGE_NAME% >nul 2>&1
)

:: Build the image (uses cache if no changes)
echo Building %IMAGE_NAME%...
docker build -t %IMAGE_NAME% "%SCRIPT_DIR%"
if errorlevel 1 (
    echo.
    echo Build failed!
    pause
    exit /b 1
)
echo Build complete!
echo.

:: Run the pipeline
echo Running pipeline...
docker run --rm -it --gpus all ^
    -v "%INPUT_DIR%:/app/input" ^
    -v "%OUTPUT_DIR%:/app/output" ^
    -v "%WEIGHTS_DIR%:/app/src/models/weights" ^
    -v dataprocess-pip-cache:/root/.cache/pip ^
    -v dataprocess-packages:/usr/local/lib/python3.10/dist-packages ^
    -v dataprocess-insightface:/root/.insightface ^
    %IMAGE_NAME%

echo.
echo Done!
pause
