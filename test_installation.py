#!/usr/bin/env python3
"""
Test script to verify that all required dependencies are installed
and the data synthesis tool is working properly.
"""

import sys
import importlib.util
import os
from datetime import datetime


def check_package(package_name):
    """Check if a Python package is installed"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def main():
    """Check dependencies and test basic functionality"""
    print("Iowa Liquor Sales Data Synthesizer - Installation Test")
    print("====================================================")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"\nPython version: {python_version}")
    major, minor, patch = map(int, python_version.split('.'))
    if major < 3 or (major == 3 and minor < 6):
        print("❌ Python 3.6 or higher is required")
        return False
    else:
        print("✓ Python version is compatible")
    
    # Check required packages
    required_packages = ["pandas", "numpy", "matplotlib", "scipy"]
    missing_packages = []
    
    print("\nChecking required packages:")
    for package in required_packages:
        if check_package(package):
            print(f"✓ {package} is installed")
        else:
            print(f"❌ {package} is not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n❌ Some required packages are missing. Install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    # Check if main script exists
    print("\nChecking for main script:")
    if os.path.exists("analyze_and_generate.py"):
        print("✓ analyze_and_generate.py found")
    else:
        print("❌ analyze_and_generate.py not found in current directory")
        return False
    
    # Check sample data
    print("\nChecking for sample data:")
    if os.path.exists("Iowa Liquor Sales 2024.csv"):
        print("✓ Sample data file found")
    else:
        print("⚠️ Sample data file 'Iowa Liquor Sales 2024.csv' not found")
        print("  You will need to provide your own CSV file and modify the scripts accordingly")
    
    # Test imports from main script
    print("\nTesting imports from main script:")
    try:
        from analyze_and_generate import analyze_csv, generate_synthetic_data
        print("✓ Successfully imported functions from analyze_and_generate.py")
    except ImportError as e:
        print(f"❌ Error importing from analyze_and_generate.py: {e}")
        return False
    except SyntaxError as e:
        print(f"❌ Syntax error in analyze_and_generate.py: {e}")
        return False
    
    print("\n✓ All installation tests passed! The tool is ready to use.")
    print("\nTo generate synthetic data, run:")
    print("  python analyze_and_generate.py")
    print("\nOr for custom parameters:")
    print("  python example.py --help")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
