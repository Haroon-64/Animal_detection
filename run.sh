#!/bin/bash

required_libraries=("numpy" "pandas" "PyQt5" "tensorflow" "tensorflow_hub")
command -v python >/dev/null 2>&1 || { echo "Python is not installed. Please install Python and try again."; exit 1; }

for library in "${required_libraries[@]}"; do
    if ! pip show "$library" >/dev/null 2>&1; then
        read -p "Library $library is missing. Do you want to install it? (y/n): " install_library
        if [ "$install_library" == "y" ]; then
            echo "Installing library: $library"
            pip install "$library"
        else
            echo "Library $library is required. Exiting."
            exit 1
        fi
    fi
done

# Execute the Python script
python main.py
