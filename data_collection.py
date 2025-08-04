import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius
import pandas as pd
import time
import os
from typing import Dict, List, Optional
import json

class MusicDataCollector:
    def __init__(self, spotify_client_id: str, spotify_client_secret: str, genius_token: str):
        """Initialize the music data collector with API credentials."""
        # Spotify setup
        client_credentials_manager = SpotifyClientCredentials(
            client_id=0ba8d059a6b74ddf893004408bd7353b,
            client_secret=6835e7a1c7e04e339c3d7e0edc660389
        )
        self.spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        
        # Genius setup
        self.genius = lyricsgenius.Genius(genius_token)
        self.genius.verbose = False
        self.genius.remove_section_headers = True
        
        # Genre mapping with time periods
        self.genre_mapping = {
            '1980s': ['rock', 'pop', 'disco'],
            '2000s': ['hip-hop', 'r&b', 'electronic'],
            '2020s': ['k-pop', 'afrobeat', 'hyperpop']
        }
    
    def get_top_tracks_by_genre(self, genre: str, decade: str, limit: int = 100) -> List[Dict]:
        """Get top tracks for a specific genre and decade."""
        tracks = []
        
        # Create search queries based on genre and decade
        year_range = self._get_year_range(decade)
        search_queries = [
            f"genre:{genre} year:{year_range}",
            f"{genre} year:{year_range}",
            f"genre:\"{genre}\" year:{year_range}"
        ]
        
        for query in search_queries:
            try:
                results = self.spotify.search(q=query, type='track', limit=50, market='US')
                for track in results['tracks']['items']:
                    if len(tracks) >= limit:
                        break
                    
                    track_info = {
                        'id': track['id'],
                        'name': track['name'],
                        'artist': track['artists'][0]['name'],
                        'album': track['album']['name'],
                        'release_date': track['album']['release_date'],
                        'popularity': track['popularity'],
                        'genre': genre,
                        'decade': decade
                    }
                    tracks.append(track_info)
                
                if len(tracks) >= limit:
                    break
                    
            except Exception as e:
                print(f"Error searching for {genre} in {decade}: {e}")
                continue
        
        return tracks[:limit]
    
    def get_audio_features(self, track_ids: List[str]) -> List[Dict]:
        """Get audio features for a list of track IDs."""
        audio_features = []
        
        # Spotify API allows max 100 tracks per request
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i+100]
            try:
                features = self.spotify.audio_features(batch)
                for feature in features:
                    if feature:  # Some tracks might not have audio features
                        audio_features.append(feature)
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"Error getting audio features: {e}")
                continue
        
        return audio_features
    
    def get_lyrics(self, track_name: str, artist_name: str) -> Optional[str]:
        """Get lyrics for a track using Genius API."""
        try:
            song = self.genius.search_song(track_name, artist_name)
            if song:
                return song.lyrics
        except Exception as e:
            print(f"Error getting lyrics for {track_name} by {artist_name}: {e}")
        return None
    
    def collect_all_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Collect all music data and return genre evolution and lyrics dataframes."""
        all_tracks = []
        all_lyrics = []
        
        for decade, genres in self.genre_mapping.items():
            print(f"Collecting data for {decade}...")
            
            for genre in genres:
                print(f"  Processing {genre}...")
                
                # Get top tracks for this genre and decade
                tracks = self.get_top_tracks_by_genre(genre, decade, 100)
                
                if not tracks:
                    print(f"    No tracks found for {genre} in {decade}")
                    continue
                
                # Get audio features
                track_ids = [track['id'] for track in tracks]
                audio_features = self.get_audio_features(track_ids)
                
                # Create a mapping of track_id to audio features
                features_map = {f['id']: f for f in audio_features if f}
                
                # Combine track info with audio features
                for track in tracks:
                    if track['id'] in features_map:
                        combined_data = {**track, **features_map[track['id']]}
                        all_tracks.append(combined_data)
                        
                        # Collect lyrics
                        lyrics = self.get_lyrics(track['name'], track['artist'])
                        if lyrics:
                            lyrics_data = {
                                'track_id': track['id'],
                                'track_name': track['name'],
                                'artist': track['artist'],
                                'genre': genre,
                                'decade': decade,
                                'lyrics': lyrics
                            }
                            all_lyrics.append(lyrics_data)
                
                time.sleep(1)  # Rate limiting between genres
        
        # Create DataFrames
        genre_df = pd.DataFrame(all_tracks)
        lyrics_df = pd.DataFrame(all_lyrics)
        
        return genre_df, lyrics_df
    
    def _get_year_range(self, decade: str) -> str:
        """Convert decade to year range for Spotify search."""
        year_ranges = {
            '1980s': '1980-1989',
            '2000s': '2000-2009',
            '2020s': '2020-2024'
        }
        return year_ranges.get(decade, '2020-2024')
    
    def save_data(self, genre_df: pd.DataFrame, lyrics_df: pd.DataFrame):
        """Save collected data to CSV files."""
        # Clean and prepare genre evolution data
        audio_feature_columns = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
        ]
        
        # Select relevant columns for genre evolution
        genre_columns = [
            'id', 'name', 'artist', 'album', 'release_date', 'popularity', 
            'genre', 'decade'
        ] + audio_feature_columns
        
        genre_evolution_df = genre_df[genre_columns].copy()
        
        # Save to CSV
        genre_evolution_df.to_csv('genre_evolution.csv', index=False)
        lyrics_df.to_csv('lyrics_corpus.csv', index=False)
        
        print(f"Saved {len(genre_evolution_df)} tracks to genre_evolution.csv")
        print(f"Saved {len(lyrics_df)} lyrics to lyrics_corpus.csv")


def main():
    """Main function to run data collection."""
    # API credentials - In a real application, these should be environment variables
    spotify_client_id = os.getenv('SPOTIFY_CLIENT_ID', 'your_spotify_client_id')
    spotify_client_secret = os.getenv('SPOTIFY_CLIENT_SECRET', 'your_spotify_client_secret')
    genius_token = os.getenv('GENIUS_TOKEN', 'your_genius_token')
    
    if any(cred in ['your_spotify_client_id', 'your_spotify_client_secret', 'your_genius_token'] 
           for cred in [spotify_client_id, spotify_client_secret, genius_token]):
        print("Please set your API credentials as environment variables:")
        print("export SPOTIFY_CLIENT_ID='your_actual_client_id'")
        print("export SPOTIFY_CLIENT_SECRET='your_actual_client_secret'")
        print("export GENIUS_TOKEN='your_actual_genius_token'")
        return
    
    # Initialize collector
    collector = MusicDataCollector(spotify_client_id, spotify_client_secret, genius_token)
    
    # Collect data
    print("Starting data collection...")
    genre_df, lyrics_df = collector.collect_all_data()
    
    # Save data
    collector.save_data(genre_df, lyrics_df)
    
    print("Data collection completed!")


if __name__ == "__main__":
    main()
