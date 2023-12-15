@echo off
setlocal enabledelayedexpansion


set "required_libraries=numpy pandas PyQt5 tensorflow tensorflow_hub"


python --version 2>nul || (
    echo Python is not installed. Please install Python and try again.
    exit /b
)


for %%i in (%required_libraries%) do (
    pip show %%i 2>nul || (
        set /p install_library="Library %%i is missing. Do you want to install it? (y/n): "
        if /i "!install_library!" equ "y" (
            echo Installing  library: %%i
            pip install %%i
        ) else (
            echo Library %%i is required. Exiting.
            exit /b
        )
    )
)

python main.py

endlocal
