#!/usr/bin/env python3
"""
Master script to run the complete Music Analytics pipeline.
This script will execute all components in the correct order.
"""

import os
import sys
import subprocess
import time
from typing import List

def run_command(command: str, description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ SUCCESS: {description}")
        if result.stdout:
            print("OUTPUT:", result.stdout[:500] + ("..." if len(result.stdout) > 500 else ""))
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: {description}")
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_api_credentials() -> bool:
    """Check if API credentials are set."""
    required_vars = ['SPOTIFY_CLIENT_ID', 'SPOTIFY_CLIENT_SECRET', 'GENIUS_TOKEN']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var) or os.getenv(var) in ['your_spotify_client_id', 'your_spotify_client_secret', 'your_genius_token']:
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ MISSING API CREDENTIALS")
        print("Please set the following environment variables:")
        for var in missing_vars:
            print(f"  export {var}='your_actual_token'")
        print("\nGet credentials from:")
        print("  Spotify: https://developer.spotify.com/dashboard/")
        print("  Genius: https://genius.com/api-clients")
        return False
    
    print("✅ API credentials found")
    return True

def check_dependencies() -> bool:
    """Check if required Python packages are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        'spotipy', 'lyricsgenius', 'pandas', 'numpy', 'plotly', 'dash',
        'scikit-learn', 'textblob', 'gensim', 'ruptures', 'wordcloud',
        'shap', 'nltk', 'seaborn', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ MISSING PACKAGES: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies found")
    return True

def main():
    """Run the complete music analytics pipeline."""
    print("🎵 MUSIC ANALYTICS PIPELINE")
    print("🎵 Comprehensive Genre Evolution & Viral Hit Analysis")
    print("🎵 " + "="*50)
    
    start_time = time.time()
    
    # Pre-flight checks
    print("\n📋 PRE-FLIGHT CHECKS")
    
    if not check_dependencies():
        sys.exit(1)
    
    if not check_api_credentials():
        print("\n⚠️  WARNING: API credentials not found.")
        print("The pipeline will create sample data files for testing.")
        print("For full functionality, please set up API credentials.")
        
        user_input = input("\nContinue with limited functionality? (y/N): ")
        if user_input.lower() != 'y':
            sys.exit(1)
    
    # Pipeline steps
    steps = [
        ("python data_collection.py", "Data Collection (Spotify + Genius APIs)"),
        ("python genre_evolution_analysis.py", "Genre Evolution Analysis"),
        ("python nlp_analysis.py", "NLP & Sentiment Analysis"),
        ("python viral_predictor.py", "Viral Hit Predictor Training"),
        ("python dashboard.py &", "Launch Interactive Dashboard")
    ]
    
    successful_steps = 0
    total_steps = len(steps)
    
    print(f"\n🚀 STARTING PIPELINE ({total_steps} steps)")
    
    for i, (command, description) in enumerate(steps, 1):
        print(f"\n[{i}/{total_steps}] Starting: {description}")
        
        # Special handling for dashboard (runs in background)
        if "dashboard.py" in command:
            print(f"\n🌐 LAUNCHING DASHBOARD")
            print("Dashboard will be available at: http://localhost:8050")
            print("Press Ctrl+C to stop all processes")
            
            try:
                subprocess.Popen(["python", "dashboard.py"])
                print("✅ Dashboard launched successfully")
                successful_steps += 1
                break  # Exit after launching dashboard
            except Exception as e:
                print(f"❌ Failed to launch dashboard: {e}")
                break
        else:
            if run_command(command, description):
                successful_steps += 1
            else:
                print(f"⚠️  Step failed, but continuing pipeline...")
                # Continue with next step even if one fails
                continue
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n🎯 PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {duration:.1f} seconds")
    print(f"Successful steps: {successful_steps}/{total_steps}")
    print(f"Success rate: {successful_steps/total_steps*100:.1f}%")
    
    if successful_steps == total_steps:
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("\n📁 Generated Files:")
        generated_files = [
            "genre_evolution.csv", "lyrics_corpus.csv", 
            "sentiment_analysis.csv", "topics_by_decade.csv",
            "viral_predictor.pkl", "genre_radar_charts.html",
            "genre_network.html", "wordclouds_by_decade.png",
            "shap_summary_plot.png", "sentiment_analysis.png"
        ]
        
        for file in generated_files:
            if os.path.exists(file):
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file} (not found)")
        
        print("\n🌐 Dashboard Access:")
        print("  URL: http://localhost:8050")
        print("  Features: Genre evolution, viral prediction, network analysis")
        
        print("\n📊 Quick Analysis Commands:")
        print("  # View feature importance")
        print("  python -c \"import pandas as pd; print(pd.read_csv('shap_feature_importance.csv').head())\"")
        print("  # Check sentiment trends")
        print("  python -c \"import pandas as pd; df=pd.read_csv('sentiment_analysis.csv'); print(df.groupby('decade')['polarity'].mean())\"")
        
    else:
        print("⚠️  PIPELINE COMPLETED WITH SOME FAILURES")
        print("Check the error messages above for details.")
        
        if successful_steps >= 2:
            print("Enough steps completed for basic analysis.")
    
    print(f"\n🎵 Thank you for using the Music Analytics System!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Pipeline interrupted by user")
        print("Partial results may be available in generated files")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()