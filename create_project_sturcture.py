import os
from pathlib import Path

def create_project_structure(root_dir):
    # Define the directory structure
    dirs = [
        ".github/workflows",
        "data/raw",
        "data/processed",
        "src/model_finetuning",
        "notebooks",
        "reports",
        "config",
        "tests",
        "models",
        "outputs",
        "photos",
    ]
    
    # Create root directory if it doesn't exist
    root_path = Path(root_dir)
    root_path.mkdir(exist_ok=True)
    
    # Create all directories
    for dir_path in dirs:
        full_path = root_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {full_path}")
    
    # Create empty files
    files = [
        "src/telegram_scraper.py",
        "src/preprocessor.py",
        "src/labeling.py",
        "notebooks/data_ingestion_eda.ipynb",
        "notebooks/preprocessing_eda.ipynb",
        "requirements.txt",
        "README.md",
        ".gitignore",
        "config/config.py",
        ".env",
    ]
    
    for file_path in files:
        full_path = root_path / file_path
        full_path.touch()
        print(f"Created file: {full_path}")

    

if __name__ == "__main__":
    project_root = "../EthioMart"
    create_project_structure(project_root)
