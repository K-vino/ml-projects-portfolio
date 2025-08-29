#!/usr/bin/env python3
"""
🚀 ML Projects Portfolio Setup Script
Automated setup for all 12 machine learning projects
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                🚀 ML Projects Portfolio Setup                ║
    ║                                                              ║
    ║  Setting up 12 Machine Learning Projects                    ║
    ║  From Beginner to Expert Level                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required!")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    try:
        # Install main requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Main packages installed successfully!")
        
        # Install additional packages for specific projects
        additional_packages = [
            "tensorflow>=2.10.0",
            "torch>=1.12.0",
            "transformers>=4.20.0",
            "plotly>=5.0.0",
            "streamlit>=1.25.0",
            "flask>=2.2.0",
            "fastapi>=0.85.0",
            "uvicorn>=0.18.0"
        ]
        
        print("\n🔧 Installing additional packages for advanced projects...")
        for package in additional_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"✅ {package.split('>=')[0]} installed")
            except subprocess.CalledProcessError:
                print(f"⚠️  {package.split('>=')[0]} installation failed (optional)")
        
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Package installation failed: {e}")
        return False

def create_project_structure():
    """Create necessary directories and files"""
    print("\n📁 Creating project structure...")
    
    projects = [
        "01-iris-classifier",
        "02-titanic-survival", 
        "03-boston-housing",
        "04-diabetes-prediction",
        "05-customer-segmentation",
        "06-fraud-detection",
        "07-movie-recommender",
        "08-mnist-cnn",
        "09-stock-prediction",
        "10-sentiment-analysis",
        "11-automl-shap",
        "12-model-deployment"
    ]
    
    for project in projects:
        project_path = Path(project)
        
        # Create main directories
        (project_path / "data").mkdir(parents=True, exist_ok=True)
        (project_path / "results").mkdir(parents=True, exist_ok=True)
        (project_path / "models").mkdir(parents=True, exist_ok=True)
        
        print(f"✅ {project} structure created")
    
    # Create additional directories for deployment project
    deployment_path = Path("12-model-deployment")
    (deployment_path / "templates").mkdir(parents=True, exist_ok=True)
    (deployment_path / "static").mkdir(parents=True, exist_ok=True)
    
    print("✅ All project structures created!")

def download_datasets():
    """Download and prepare datasets"""
    print("\n📊 Preparing datasets...")
    
    try:
        # Most datasets are built into scikit-learn or will be generated
        # This function can be extended to download external datasets
        
        print("✅ Built-in datasets will be loaded automatically")
        print("✅ Sample datasets will be generated in notebooks")
        
        # Create a datasets info file
        datasets_info = {
            "01-iris-classifier": "Built-in sklearn dataset",
            "02-titanic-survival": "Built-in seaborn dataset", 
            "03-boston-housing": "Built-in sklearn dataset",
            "04-diabetes-prediction": "Generated sample data",
            "05-customer-segmentation": "Generated sample data",
            "06-fraud-detection": "Kaggle dataset (download manually)",
            "07-movie-recommender": "MovieLens dataset (download manually)",
            "08-mnist-cnn": "Built-in tensorflow dataset",
            "09-stock-prediction": "Yahoo Finance API (automatic)",
            "10-sentiment-analysis": "Built-in tensorflow dataset",
            "11-automl-shap": "Multiple built-in datasets",
            "12-model-deployment": "Uses trained models from other projects"
        }
        
        with open("datasets_info.json", "w") as f:
            json.dump(datasets_info, f, indent=2)
        
        return True
    
    except Exception as e:
        print(f"⚠️  Dataset preparation warning: {e}")
        return True  # Non-critical

def create_run_scripts():
    """Create convenient run scripts"""
    print("\n📝 Creating run scripts...")
    
    # Create Jupyter launcher script
    jupyter_script = """#!/bin/bash
# Launch Jupyter Notebook for ML Projects Portfolio

echo "🚀 Starting Jupyter Notebook for ML Projects Portfolio..."
echo "📂 Navigate to any project folder and open the notebook.ipynb file"
echo "🌐 Jupyter will open in your default browser"
echo ""

jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
"""
    
    with open("start_jupyter.sh", "w") as f:
        f.write(jupyter_script)
    
    # Make executable on Unix systems
    try:
        os.chmod("start_jupyter.sh", 0o755)
    except:
        pass  # Windows doesn't need this
    
    # Create deployment launcher script
    deployment_script = """#!/bin/bash
# Launch ML Model Deployment Examples

echo "🚀 ML Model Deployment Options:"
echo ""
echo "1. Flask API:      python 12-model-deployment/app.py"
echo "2. Streamlit App:  streamlit run 12-model-deployment/streamlit_app.py"
echo "3. Docker Build:   docker build -t ml-api 12-model-deployment/"
echo ""
echo "Choose your deployment method and run the appropriate command!"
"""
    
    with open("start_deployment.sh", "w") as f:
        f.write(deployment_script)
    
    try:
        os.chmod("start_deployment.sh", 0o755)
    except:
        pass
    
    print("✅ Run scripts created!")

def run_tests():
    """Run basic tests to verify setup"""
    print("\n🧪 Running setup verification tests...")
    
    try:
        # Test imports
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import sklearn
        
        print("✅ Core data science libraries working")
        
        # Test optional imports
        try:
            import tensorflow as tf
            print("✅ TensorFlow available")
        except ImportError:
            print("⚠️  TensorFlow not available (optional for advanced projects)")
        
        try:
            import streamlit as st
            print("✅ Streamlit available")
        except ImportError:
            print("⚠️  Streamlit not available (optional for deployment)")
        
        try:
            import flask
            print("✅ Flask available")
        except ImportError:
            print("⚠️  Flask not available (optional for deployment)")
        
        return True
    
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    next_steps = """
    🎉 Setup Complete! Here's how to get started:

    📚 LEARNING PATH:
    
    🟢 Beginner Projects (Start Here):
    1. cd 01-iris-classifier && jupyter notebook notebook.ipynb
    2. cd 02-titanic-survival && jupyter notebook notebook.ipynb  
    3. cd 03-boston-housing && jupyter notebook notebook.ipynb

    🟡 Intermediate Projects:
    4. cd 04-diabetes-prediction && jupyter notebook notebook.ipynb
    5. cd 05-customer-segmentation && jupyter notebook notebook.ipynb
    6. cd 06-fraud-detection && jupyter notebook notebook.ipynb
    7. cd 07-movie-recommender && jupyter notebook notebook.ipynb

    🔴 Advanced Projects:
    8. cd 08-mnist-cnn && jupyter notebook notebook.ipynb
    9. cd 09-stock-prediction && jupyter notebook notebook.ipynb
    10. cd 10-sentiment-analysis && jupyter notebook notebook.ipynb

    ⚫ Expert Projects:
    11. cd 11-automl-shap && jupyter notebook notebook.ipynb
    12. cd 12-model-deployment && jupyter notebook notebook.ipynb

    🚀 QUICK START OPTIONS:
    
    • Launch Jupyter:     ./start_jupyter.sh (or jupyter notebook)
    • Try Deployment:     ./start_deployment.sh
    • Flask API:          python 12-model-deployment/app.py
    • Streamlit App:      streamlit run 12-model-deployment/streamlit_app.py

    📖 DOCUMENTATION:
    
    • Each project has a detailed README.md
    • Notebooks include step-by-step explanations
    • Check datasets_info.json for data sources

    🎯 TIPS:
    
    • Start with beginner projects to build fundamentals
    • Each project builds on previous concepts
    • Focus on understanding before moving to next level
    • Use the deployment project to showcase your work

    Happy Learning! 🚀
    """
    print(next_steps)

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("⚠️  Some packages failed to install. You may need to install them manually.")
    
    # Create project structure
    create_project_structure()
    
    # Prepare datasets
    download_datasets()
    
    # Create run scripts
    create_run_scripts()
    
    # Run tests
    if not run_tests():
        print("⚠️  Some tests failed. Check your installation.")
    
    # Print next steps
    print_next_steps()
    
    print("🎉 Setup completed successfully!")

if __name__ == "__main__":
    main()
