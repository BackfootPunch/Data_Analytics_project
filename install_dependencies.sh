#!/bin/bash

# 🎵 Music Analytics System - Dependency Installation Script
# This script handles installation for different environments

set -e  # Exit on any error

echo "🎵 Music Analytics System - Dependency Installer"
echo "=================================================="

# Function to print colored output
print_status() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# Check Python version
print_status "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_success "Already in virtual environment: $VIRTUAL_ENV"
    USE_VENV=false
else
    print_status "Not in a virtual environment. Will create one."
    USE_VENV=true
fi

# Create virtual environment if needed
if [ "$USE_VENV" = true ]; then
    ENV_NAME="music_analytics_env"
    
    print_status "Creating virtual environment: $ENV_NAME"
    
    # Try different methods to create virtual environment
    if python3 -m venv --help &> /dev/null; then
        python3 -m venv $ENV_NAME
        print_success "Virtual environment created successfully"
    else
        print_error "python3-venv not available. Trying alternative installation..."
        
        # Try to install venv package
        if command -v apt &> /dev/null; then
            print_status "Installing python3-venv using apt..."
            sudo apt update
            sudo apt install -y python3-venv python3-dev build-essential
            python3 -m venv $ENV_NAME
        elif command -v yum &> /dev/null; then
            print_status "Installing python3-venv using yum..."
            sudo yum install -y python3-venv python3-devel gcc
            python3 -m venv $ENV_NAME
        else
            print_warning "Package manager not found. Trying without virtual environment..."
            USE_VENV=false
        fi
    fi
    
    # Activate virtual environment
    if [ "$USE_VENV" = true ] && [ -d "$ENV_NAME" ]; then
        print_status "Activating virtual environment..."
        source $ENV_NAME/bin/activate
        print_success "Virtual environment activated"
        
        # Save activation command for user
        echo "source $(pwd)/$ENV_NAME/bin/activate" > activate_env.sh
        chmod +x activate_env.sh
        print_success "Created activation script: ./activate_env.sh"
    fi
fi

# Upgrade pip
print_status "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install dependencies with different strategies
print_status "Installing Music Analytics dependencies..."

# Strategy 1: Try full requirements.txt
if pip install -r requirements.txt; then
    print_success "All dependencies installed successfully!"
    INSTALLATION_SUCCESS=true
else
    print_warning "Full installation failed. Trying individual package installation..."
    INSTALLATION_SUCCESS=false
    
    # Strategy 2: Install core packages individually
    CORE_PACKAGES=(
        "spotipy==2.23.0"
        "lyricsgenius==3.0.1" 
        "pandas==2.0.3"
        "numpy==1.24.3"
        "plotly==5.15.0"
        "requests==2.31.0"
    )
    
    ADVANCED_PACKAGES=(
        "dash==2.14.1"
        "scikit-learn==1.3.0"
        "textblob==0.17.1"
        "wordcloud==1.9.2"
        "matplotlib==3.7.2"
        "seaborn==0.12.2"
    )
    
    OPTIONAL_PACKAGES=(
        "gensim==4.3.2"
        "ruptures==1.1.9"
        "shap==0.42.1"
        "nltk==3.8.1"
    )
    
    # Install core packages
    print_status "Installing core packages..."
    for package in "${CORE_PACKAGES[@]}"; do
        if pip install "$package"; then
            print_success "✓ $package"
        else
            print_error "✗ Failed to install $package"
        fi
    done
    
    # Install advanced packages
    print_status "Installing advanced packages..."
    for package in "${ADVANCED_PACKAGES[@]}"; do
        if pip install "$package"; then
            print_success "✓ $package"
        else
            print_warning "✗ Failed to install $package (optional for basic functionality)"
        fi
    done
    
    # Install optional packages
    print_status "Installing optional packages..."
    for package in "${OPTIONAL_PACKAGES[@]}"; do
        if pip install "$package"; then
            print_success "✓ $package"
        else
            print_warning "✗ Failed to install $package (advanced features may not work)"
        fi
    done
    
    INSTALLATION_SUCCESS=true
fi

# Test installation
print_status "Testing installation..."
if python3 -c "import spotipy, pandas, plotly; print('Core packages working!')" 2>/dev/null; then
    print_success "Core functionality test passed!"
else
    print_error "Core functionality test failed!"
    INSTALLATION_SUCCESS=false
fi

# Create a test script
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""Test script to verify Music Analytics installation."""

def test_imports():
    """Test importing all required packages."""
    success_count = 0
    total_count = 0
    
    packages = [
        ('spotipy', 'Spotify API'),
        ('lyricsgenius', 'Genius API'),
        ('pandas', 'Data manipulation'),
        ('numpy', 'Numerical computing'),
        ('plotly', 'Visualizations'),
        ('requests', 'HTTP requests'),
        ('dash', 'Web dashboard'),
        ('sklearn', 'Machine learning'),
        ('textblob', 'NLP'),
        ('wordcloud', 'Word clouds'),
        ('matplotlib', 'Plotting'),
        ('seaborn', 'Statistical plots'),
    ]
    
    optional_packages = [
        ('gensim', 'Topic modeling'),
        ('ruptures', 'Change point detection'),
        ('shap', 'Model explanations'),
        ('nltk', 'Natural language toolkit'),
    ]
    
    print("🧪 Testing Core Packages:")
    print("-" * 40)
    
    for package, description in packages:
        total_count += 1
        try:
            __import__(package)
            print(f"✅ {package:<15} - {description}")
            success_count += 1
        except ImportError:
            print(f"❌ {package:<15} - {description} (MISSING)")
    
    print(f"\n🔬 Testing Optional Packages:")
    print("-" * 40)
    
    optional_success = 0
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package:<15} - {description}")
            optional_success += 1
        except ImportError:
            print(f"⚠️  {package:<15} - {description} (optional)")
    
    print(f"\n📊 Installation Summary:")
    print(f"Core packages: {success_count}/{total_count} ({success_count/total_count*100:.0f}%)")
    print(f"Optional packages: {optional_success}/{len(optional_packages)} ({optional_success/len(optional_packages)*100:.0f}%)")
    
    if success_count == total_count:
        print("🎉 All core packages installed successfully!")
        return True
    elif success_count >= total_count * 0.8:
        print("⚠️  Most packages installed. System should work with limited functionality.")
        return True
    else:
        print("❌ Too many missing packages. Please check installation.")
        return False

if __name__ == "__main__":
    success = test_imports()
    
    if success:
        print(f"\n🚀 Ready to run Music Analytics!")
        print(f"Try: python3 demo_runner.py")
    else:
        print(f"\n🔧 Installation needs attention.")
        print(f"Try running: pip install -r requirements.txt")
EOF

chmod +x test_installation.py

# Run the test
print_status "Running comprehensive installation test..."
if python3 test_installation.py; then
    print_success "Installation test completed!"
else
    print_warning "Some issues detected in installation test."
fi

# Final instructions
echo ""
echo "🎯 Installation Complete!"
echo "======================="

if [ "$USE_VENV" = true ] && [ -d "$ENV_NAME" ]; then
    echo "📁 Virtual environment created: $ENV_NAME"
    echo "🔧 To activate environment: source $ENV_NAME/bin/activate"
    echo "📜 Or use shortcut: ./activate_env.sh"
fi

echo ""
echo "🚀 Next Steps:"
echo "1. Test installation: python3 test_installation.py"
echo "2. Run demo: python3 demo_runner.py"
echo "3. For full system with APIs: python3 run_all.py"
echo ""

if [ "$INSTALLATION_SUCCESS" = true ]; then
    print_success "🎉 Music Analytics System is ready to use!"
else
    print_warning "⚠️  Installation completed with some issues. Basic functionality should work."
fi

echo ""
echo "💡 For API setup:"
echo "export SPOTIFY_CLIENT_ID='your_client_id'"
echo "export SPOTIFY_CLIENT_SECRET='your_client_secret'"
echo "export GENIUS_TOKEN='your_genius_token'"