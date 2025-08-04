# 📦 Installation Guide for Music Analytics System

This guide covers how to install dependencies for the Music Analytics System on different operating systems and environments.

## 🚀 Quick Start (Recommended)

### Option 1: Virtual Environment (Recommended)

The safest way to install dependencies is using a Python virtual environment:

```bash
# Navigate to project directory
cd /path/to/music-analytics

# Create virtual environment
python3 -m venv music_analytics_env

# Activate virtual environment
# On Linux/Mac:
source music_analytics_env/bin/activate
# On Windows:
music_analytics_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using pipx (For Isolated Applications)

```bash
# Install pipx if not already installed
python3 -m pip install --user pipx
python3 -m pipx ensurepath

# Install each package independently
pipx install spotipy
pipx install lyricsgenius
pipx install pandas
# ... (continue for other packages)
```

## 🖥️ System-Specific Installation

### 🐧 Linux (Ubuntu/Debian)

1. **Update package manager:**
```bash
sudo apt update
sudo apt upgrade
```

2. **Install Python development tools:**
```bash
sudo apt install python3 python3-pip python3-venv python3-dev
sudo apt install build-essential
```

3. **Create and activate virtual environment:**
```bash
python3 -m venv music_analytics_env
source music_analytics_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 🍎 macOS

1. **Install Homebrew (if not installed):**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. **Install Python:**
```bash
brew install python3
```

3. **Create virtual environment:**
```bash
python3 -m venv music_analytics_env
source music_analytics_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 🪟 Windows

1. **Install Python from python.org:**
   - Download from: https://www.python.org/downloads/
   - Make sure to check "Add Python to PATH"

2. **Open Command Prompt or PowerShell:**
```cmd
# Create virtual environment
python -m venv music_analytics_env

# Activate virtual environment
music_analytics_env\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 🐳 Docker Installation (Advanced)

If you prefer containerized deployment:

1. **Create Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for dashboard
EXPOSE 8050

# Run the application
CMD ["python", "run_all.py"]
```

2. **Build and run:**
```bash
docker build -t music-analytics .
docker run -p 8050:8050 -e SPOTIFY_CLIENT_ID=your_id -e SPOTIFY_CLIENT_SECRET=your_secret -e GENIUS_TOKEN=your_token music-analytics
```

## 🔧 Conda Installation

If you use Anaconda/Miniconda:

```bash
# Create conda environment
conda create -n music_analytics python=3.11

# Activate environment
conda activate music_analytics

# Install packages (some via conda, others via pip)
conda install pandas numpy matplotlib seaborn scikit-learn

# Install remaining packages via pip
pip install spotipy lyricsgenius plotly dash textblob gensim ruptures wordcloud shap nltk
```

## ⚠️ Troubleshooting Common Issues

### Issue 1: "externally-managed-environment" Error

If you get this error on Ubuntu/Debian systems:

```bash
# Solution 1: Use virtual environment (recommended)
python3 -m venv music_analytics_env
source music_analytics_env/bin/activate
pip install -r requirements.txt

# Solution 2: Use system package manager
sudo apt install python3-pandas python3-numpy python3-matplotlib

# Solution 3: Override (not recommended)
pip install --break-system-packages -r requirements.txt
```

### Issue 2: Missing System Dependencies

For building certain packages, you might need:

**Linux:**
```bash
sudo apt install python3-dev build-essential libffi-dev libssl-dev
```

**macOS:**
```bash
xcode-select --install
```

### Issue 3: Memory Issues During Installation

If installation fails due to memory constraints:

```bash
# Install packages one by one
pip install spotipy
pip install pandas
pip install numpy
# ... continue individually
```

### Issue 4: Permission Errors

```bash
# Use user installation
pip install --user -r requirements.txt

# Or fix permissions (Linux/Mac)
sudo chown -R $USER ~/.local
```

## 📋 Dependency List Explanation

Here's what each major dependency does:

| Package | Purpose | Size |
|---------|---------|------|
| `spotipy` | Spotify API integration | ~50KB |
| `lyricsgenius` | Genius API for lyrics | ~100KB |
| `pandas` | Data manipulation | ~50MB |
| `numpy` | Numerical computing | ~20MB |
| `plotly` | Interactive visualizations | ~30MB |
| `dash` | Web dashboard framework | ~10MB |
| `scikit-learn` | Machine learning | ~30MB |
| `textblob` | Natural language processing | ~5MB |
| `gensim` | Topic modeling | ~50MB |
| `ruptures` | Change point detection | ~2MB |
| `wordcloud` | Word cloud generation | ~5MB |
| `shap` | Model explainability | ~10MB |
| `nltk` | NLP toolkit | ~20MB |

**Total approximate size: ~232MB**

## 🧪 Testing Installation

After installation, test that everything works:

```bash
# Activate your environment
source music_analytics_env/bin/activate  # Linux/Mac
# OR
music_analytics_env\Scripts\activate     # Windows

# Test imports
python3 -c "
import spotipy
import pandas as pd
import plotly
import dash
import sklearn
import textblob
import gensim
import ruptures
import wordcloud
import shap
import nltk
print('✅ All dependencies installed successfully!')
"

# Run demo (works without API keys)
python3 demo_runner.py
```

## 🌐 Cloud Installation

### Google Colab
```python
# In a Colab notebook cell:
!pip install spotipy lyricsgenius plotly dash scikit-learn textblob gensim ruptures wordcloud shap nltk
```

### AWS EC2 / Azure / GCP
```bash
# Connect to your instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 -m venv music_analytics_env
source music_analytics_env/bin/activate
pip install -r requirements.txt
```

## 🔄 Environment Management

### Activating Environment

**Every time you work on the project:**

```bash
# Linux/Mac
source music_analytics_env/bin/activate

# Windows
music_analytics_env\Scripts\activate

# When activated, your prompt should show: (music_analytics_env)
```

### Deactivating Environment

```bash
deactivate
```

### Updating Dependencies

```bash
# Activate environment first
source music_analytics_env/bin/activate

# Update all packages
pip install --upgrade -r requirements.txt

# Or update individually
pip install --upgrade pandas plotly scikit-learn
```

## 📚 Alternative: Minimal Installation

If you want to run just the demo without heavy ML libraries:

Create `requirements_minimal.txt`:
```
spotipy==2.23.0
lyricsgenius==3.0.1
pandas==2.0.3
plotly==5.15.0
requests==2.31.0
```

Then:
```bash
pip install -r requirements_minimal.txt
python3 demo_runner.py
```

## 🎯 Next Steps After Installation

1. **Set up API credentials:**
```bash
export SPOTIFY_CLIENT_ID='your_spotify_client_id'
export SPOTIFY_CLIENT_SECRET='your_spotify_client_secret'
export GENIUS_TOKEN='your_genius_token'
```

2. **Run the full system:**
```bash
python3 run_all.py
```

3. **Or run components individually:**
```bash
python3 data_collection.py
python3 genre_evolution_analysis.py
python3 nlp_analysis.py
python3 viral_predictor.py
python3 dashboard.py
```

## 🆘 Getting Help

If you encounter issues:

1. Check Python version: `python3 --version` (should be 3.8+)
2. Check pip version: `pip --version`
3. Try installing in a fresh virtual environment
4. Check system-specific solutions above
5. For package-specific issues, check their documentation:
   - Pandas: https://pandas.pydata.org/docs/
   - Plotly: https://plotly.com/python/
   - Scikit-learn: https://scikit-learn.org/

Remember: Virtual environments are your friend! They prevent conflicts and make dependency management much easier. 🐍✨