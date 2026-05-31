from setuptools import setup, find_packages
from pathlib import Path
 
# Read long description from README
_here = Path(__file__).parent
long_description = (_here / "README.md").read_text(encoding="utf-8") \
    if (_here / "README.md").exists() else ""
 
setup(
    name="pan-india-aq-engine",
    version="0.1.0",
    description=(
        "Pan-India Geospatial Air Quality Ingestion & Forecasting Engine — "
        "satellite + weather data fusion with CPCB AQI computation and LSTM forecasting."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Prabhav",
    python_requires=">=3.11,<3.12",  # Pin to 3.11 (see README for rationale)
 
    # Discovers all packages with an __init__.py
    packages=find_packages(exclude=["tests*", "logs*"]),
 
    # Runtime dependencies (minimal; full stack is in environment.yml)
    install_requires=[
        "numpy>=1.26,<2.0",
        "pandas>=2.1,<3.0",
        "requests>=2.31",
        "python-dotenv>=1.0",
        "loguru>=0.7",
        "tenacity>=8.2",
        "sqlalchemy>=2.0",
        "scikit-learn>=1.4",
        "xgboost>=2.0",
        "joblib>=1.3",
        "fastapi>=0.110",
        "pydantic>=2.5",
        "uvicorn[standard]>=0.27",
        "streamlit>=1.32",
        "plotly>=5.20",
    ],
 
    extras_require={
        # Install with: pip install -e ".[dev]"
        "dev": [
            "pytest>=8.0",
            "pytest-asyncio>=0.23",
            "httpx>=0.27",
            "black>=24.0",
            "isort>=5.13",
            "flake8>=7.0",
        ],
        # Install with: pip install -e ".[gee]"
        "gee": [
            "earthengine-api>=0.1.380",
        ],
        # Install with: pip install -e ".[torch]"
        "torch": [
            "torch>=2.2",
        ],
    },
 
    entry_points={
        "console_scripts": [
            # Run Module 1 pipeline from the command line:
            # aq-ingest "Hyderabad, Telangana" --mock-satellite
            "aq-ingest=ingestion.pipeline:main",
        ],
    },
 
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
 
    include_package_data=True,
    package_data={
        "": ["*.sql", "*.json", "*.yml"],
    },
)