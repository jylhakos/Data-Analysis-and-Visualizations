#!/usr/bin/env python3
"""
Project Verification Script
This script verifies that all essential files are present and properly configured
for the PowerBI Fish Weight Prediction project.
"""

import os
import sys

def check_file_exists(file_path, description):
    """Check if a file exists and is not empty"""
    if os.path.exists(file_path):
        if os.path.getsize(file_path) > 0:
            print(f"✓ {description}: {file_path}")
            return True
        else:
            print(f"⚠ {description}: {file_path} (exists but empty)")
            return False
    else:
        print(f"❌ {description}: {file_path} (missing)")
        return False

def main():
    """Main verification function"""
    print("PowerBI Fish Weight Prediction Project Verification")
    print("=" * 60)
    
    # Track verification results
    all_checks_passed = True
    
    print("\nEssential Files Check:")
    print("-" * 30)
    
    # Essential files to check
    essential_files = [
        ("requirements.txt", "Main requirements file"),
        ("setup.sh", "Setup script"),
        ("powerbi_integration.py", "PowerBI integration script"),
        ("scikit-learn/fish_predictive_model.py", "Fish prediction model"),
        ("Apache Airflow/fish_prediction_dag.py", "Airflow DAG"),
        (".gitignore", "Git ignore file"),
    ]
    
    for file_path, description in essential_files:
        if not check_file_exists(file_path, description):
            all_checks_passed = False
    
    print("\nData Files Check:")
    print("-" * 25)
    
    # Data files
    data_files = [
        ("data/Fish.csv", "Fish dataset in data directory"),
        ("Dataset/Fish.csv", "Fish dataset in Dataset directory"),
    ]
    
    data_found = False
    for file_path, description in data_files:
        if check_file_exists(file_path, description):
            data_found = True
    
    if not data_found:
        print("⚠ Warning: No Fish.csv found in data/ or Dataset/ directories")
        all_checks_passed = False
    
    print("\n" + "=" * 60)
    
    if all_checks_passed:
        print("🎉 All essential files are present!")
        print("\nNext Steps:")
        print("1. Run: ./setup.sh (to set up environment)")
        print("2. Run: source venv/bin/activate (to activate environment)")
        print("3. Run: python scikit-learn/fish_predictive_model.py (to train model)")
        print("4. Run: python powerbi_integration.py (to prepare PowerBI data)")
    else:
        print("❌ Some essential files are missing!")
        print("\nRecommended Actions:")
        print("1. Ensure all required files are present")
        print("2. Run setup.sh to create missing directories and files")
        print("3. Check that Fish.csv dataset is available")
        
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
