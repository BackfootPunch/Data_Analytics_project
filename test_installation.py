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
