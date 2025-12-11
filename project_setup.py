import subprocess
import sys

def install(package):
    """Installs a Python package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("Installing required libraries...")

# Core Data Science libraries
install('pandas')
install('numpy')
install('scikit-learn')

# Visualization
install('matplotlib')
install('seaborn')

# Machine Learning Framework
install('xgboost')

# ML Operations (Tracking)
install('mlflow')

# Deployment (User Interface)
install('streamlit')

print("Installation complete. You can now run 'data_processor.py'.")