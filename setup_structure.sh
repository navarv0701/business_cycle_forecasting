# In your business_cycle_forecasting folder, run these commands:

# Create all directories
mkdir -p bloomberg_extraction
mkdir -p data/bloomberg_cache data/processed data/sample_data
mkdir -p src/config src/data src/models src/utils src/analysis
mkdir -p notebooks
mkdir -p outputs/forecasts outputs/reports outputs/charts
mkdir -p tests

# Create all __init__.py files
touch src/__init__.py
touch src/config/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/utils/__init__.py
touch src/analysis/__init__.py
touch tests/__init__.py

# Create main files
touch main.py
touch setup.py
touch requirements.txt
touch requirements_bloomberg.txt
touch .gitignore
touch README.md

echo "Project structure created successfully!"

