import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import ruptures as rpt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class GenreEvolutionAnalyzer:
    def __init__(self, data_file: str = 'genre_evolution.csv'):
        """Initialize the analyzer with genre evolution data."""
        self.df = pd.read_csv(data_file)
        self.audio_features = [
            'danceability', 'energy', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo', 'loudness'
        ]
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare and clean the data for analysis."""
        # Convert release_date to datetime
        self.df['release_date'] = pd.to_datetime(self.df['release_date'])
        self.df['year'] = self.df['release_date'].dt.year
        
        # Normalize tempo and loudness to 0-1 scale
        scaler = StandardScaler()
        normalized_features = ['tempo', 'loudness']
        
        for feature in normalized_features:
            if feature in self.df.columns:
                self.df[f'{feature}_normalized'] = scaler.fit_transform(self.df[[feature]])
                # Convert to 0-1 scale
                self.df[f'{feature}_normalized'] = (
                    (self.df[f'{feature}_normalized'] - self.df[f'{feature}_normalized'].min()) /
                    (self.df[f'{feature}_normalized'].max() - self.df[f'{feature}_normalized'].min())
                )
        
        # Update audio features list to use normalized versions
        self.audio_features = [
            'danceability', 'energy', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo_normalized', 'loudness_normalized'
        ]
    
    def calculate_decade_changes(self) -> pd.DataFrame:
        """Calculate decade-by-decade changes in audio features."""
        decade_stats = []
        
        for decade in self.df['decade'].unique():
            decade_data = self.df[self.df['decade'] == decade]
            
            for genre in decade_data['genre'].unique():
                genre_data = decade_data[decade_data['genre'] == genre]
                
                if len(genre_data) > 0:
                    stats = {
                        'decade': decade,
                        'genre': genre,
                        'track_count': len(genre_data),
                        'avg_popularity': genre_data['popularity'].mean()
                    }
                    
                    # Calculate mean and std for each audio feature
                    for feature in self.audio_features:
                        if feature in genre_data.columns:
                            stats[f'{feature}_mean'] = genre_data[feature].mean()
                            stats[f'{feature}_std'] = genre_data[feature].std()
                    
                    decade_stats.append(stats)
        
        decade_df = pd.DataFrame(decade_stats)
        
        # Calculate changes between decades for each genre
        changes = []
        for genre in decade_df['genre'].unique():
            genre_data = decade_df[decade_df['genre'] == genre].sort_values('decade')
            
            for i in range(1, len(genre_data)):
                current = genre_data.iloc[i]
                previous = genre_data.iloc[i-1]
                
                change_record = {
                    'genre': genre,
                    'from_decade': previous['decade'],
                    'to_decade': current['decade']
                }
                
                for feature in self.audio_features:
                    mean_col = f'{feature}_mean'
                    if mean_col in current.index and mean_col in previous.index:
                        change = current[mean_col] - previous[mean_col]
                        change_record[f'{feature}_change'] = change
                        change_record[f'{feature}_percent_change'] = (
                            change / previous[mean_col] * 100 if previous[mean_col] != 0 else 0
                        )
                
                changes.append(change_record)
        
        self.decade_stats = decade_df
        self.decade_changes = pd.DataFrame(changes)
        
        return decade_df
    
    def detect_change_points(self, genre: str) -> Dict[str, List[int]]:
        """Detect change points in audio features for a specific genre."""
        genre_data = self.df[self.df['genre'] == genre].sort_values('year')
        
        if len(genre_data) < 10:
            return {}
        
        change_points = {}
        
        for feature in self.audio_features:
            if feature in genre_data.columns:
                # Prepare time series data
                signal = genre_data[feature].values
                
                # Use Pelt algorithm for change point detection
                algo = rpt.Pelt(model="rbf").fit(signal)
                result = algo.predict(pen=10)
                
                # Convert indices to years
                years = genre_data['year'].values
                change_point_years = [years[i-1] for i in result[:-1]]  # Exclude the last point
                
                change_points[feature] = change_point_years
        
        return change_points
    
    def create_radar_charts(self) -> go.Figure:
        """Create radar charts comparing feature profiles across decades."""
        # Calculate mean features by decade and genre
        radar_data = []
        
        for decade in self.df['decade'].unique():
            decade_data = self.df[self.df['decade'] == decade]
            
            for genre in decade_data['genre'].unique():
                genre_data = decade_data[decade_data['genre'] == genre]
                
                if len(genre_data) > 0:
                    profile = {
                        'decade': decade,
                        'genre': genre
                    }
                    
                    for feature in self.audio_features:
                        if feature in genre_data.columns:
                            profile[feature] = genre_data[feature].mean()
                    
                    radar_data.append(profile)
        
        radar_df = pd.DataFrame(radar_data)
        
        # Create subplots for each decade
        decades = sorted(self.df['decade'].unique())
        fig = make_subplots(
            rows=1, cols=len(decades),
            subplot_titles=decades,
            specs=[[{"type": "polar"}] * len(decades)]
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, decade in enumerate(decades):
            decade_data = radar_df[radar_df['decade'] == decade]
            
            for j, genre in enumerate(decade_data['genre'].unique()):
                genre_data = decade_data[decade_data['genre'] == genre]
                
                if len(genre_data) > 0:
                    values = [genre_data[feature].iloc[0] for feature in self.audio_features]
                    
                    fig.add_trace(
                        go.Scatterpolar(
                            r=values,
                            theta=self.audio_features,
                            fill='toself',
                            name=f"{genre} ({decade})",
                            line_color=colors[j % len(colors)],
                            showlegend=(i == 0)  # Only show legend for first subplot
                        ),
                        row=1, col=i+1
                    )
        
        fig.update_layout(
            title="Genre Audio Feature Profiles Across Decades",
            height=500
        )
        
        return fig
    
    def create_genre_network(self) -> go.Figure:
        """Create a network graph showing genre relationships based on feature similarity."""
        # Calculate genre profiles (mean features)
        genre_profiles = []
        
        for decade in self.df['decade'].unique():
            decade_data = self.df[self.df['decade'] == decade]
            
            for genre in decade_data['genre'].unique():
                genre_data = decade_data[decade_data['genre'] == genre]
                
                if len(genre_data) > 0:
                    profile = {
                        'node_id': f"{genre}_{decade}",
                        'genre': genre,
                        'decade': decade,
                        'popularity': genre_data['popularity'].mean(),
                        'track_count': len(genre_data)
                    }
                    
                    for feature in self.audio_features:
                        if feature in genre_data.columns:
                            profile[feature] = genre_data[feature].mean()
                    
                    genre_profiles.append(profile)
        
        profiles_df = pd.DataFrame(genre_profiles)
        
        # Calculate similarity matrix
        feature_matrix = profiles_df[self.audio_features].values
        similarity_matrix = cosine_similarity(feature_matrix)
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for idx, row in profiles_df.iterrows():
            G.add_node(
                row['node_id'],
                genre=row['genre'],
                decade=row['decade'],
                popularity=row['popularity'],
                track_count=row['track_count']
            )
        
        # Add edges based on similarity threshold
        threshold = 0.8
        for i in range(len(similarity_matrix)):
            for j in range(i+1, len(similarity_matrix)):
                if similarity_matrix[i][j] > threshold:
                    G.add_edge(
                        profiles_df.iloc[i]['node_id'],
                        profiles_df.iloc[j]['node_id'],
                        weight=similarity_matrix[i][j]
                    )
        
        # Generate layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Prepare data for Plotly
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            weight = G[edge[0]][edge[1]]['weight']
            edge_info.append(f"Similarity: {weight:.3f}")
        
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        color_map = {'1980s': 'red', '2000s': 'blue', '2020s': 'green'}
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_info = G.nodes[node]
            node_text.append(f"{node_info['genre']}<br>{node_info['decade']}<br>"
                           f"Popularity: {node_info['popularity']:.1f}<br>"
                           f"Tracks: {node_info['track_count']}")
            
            # Node size based on track count
            node_size.append(max(10, node_info['track_count'] / 2))
            node_color.append(color_map.get(node_info['decade'], 'gray'))
        
        # Create the plot
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='lightgray'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            ),
            text=[G.nodes[node]['genre'] for node in G.nodes()],
            textposition="middle center",
            showlegend=False
        ))
        
        fig.update_layout(
            title="Genre Relationship Network<br><sub>Node size = Track count, Color = Decade, Edges = Feature similarity > 0.8</sub>",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Red: 1980s, Blue: 2000s, Green: 2020s",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def generate_all_visualizations(self) -> Dict[str, go.Figure]:
        """Generate all visualizations and return as dictionary."""
        print("Calculating decade-by-decade changes...")
        self.calculate_decade_changes()
        
        print("Creating radar charts...")
        radar_fig = self.create_radar_charts()
        
        print("Creating network graph...")
        network_fig = self.create_genre_network()
        
        # Save visualizations as HTML
        radar_fig.write_html('genre_radar_charts.html')
        network_fig.write_html('genre_network.html')
        
        print("Visualizations saved as HTML files.")
        
        return {
            'radar_charts': radar_fig,
            'network_graph': network_fig
        }
    
    def save_analysis_results(self):
        """Save analysis results to CSV files."""
        if hasattr(self, 'decade_stats'):
            self.decade_stats.to_csv('decade_statistics.csv', index=False)
            print("Saved decade statistics to decade_statistics.csv")
        
        if hasattr(self, 'decade_changes'):
            self.decade_changes.to_csv('decade_changes.csv', index=False)
            print("Saved decade changes to decade_changes.csv")
        
        # Save change points for each genre
        change_points_data = []
        for genre in self.df['genre'].unique():
            cps = self.detect_change_points(genre)
            for feature, years in cps.items():
                for year in years:
                    change_points_data.append({
                        'genre': genre,
                        'feature': feature,
                        'change_point_year': year
                    })
        
        if change_points_data:
            change_points_df = pd.DataFrame(change_points_data)
            change_points_df.to_csv('change_points.csv', index=False)
            print("Saved change points to change_points.csv")


def main():
    """Main function to run genre evolution analysis."""
    try:
        # Initialize analyzer
        analyzer = GenreEvolutionAnalyzer()
        
        # Generate all visualizations
        visualizations = analyzer.generate_all_visualizations()
        
        # Save analysis results
        analyzer.save_analysis_results()
        
        print("Genre evolution analysis completed!")
        print("Generated files:")
        print("- genre_radar_charts.html")
        print("- genre_network.html")
        print("- decade_statistics.csv")
        print("- decade_changes.csv")
        print("- change_points.csv")
        
    except FileNotFoundError:
        print("Error: genre_evolution.csv not found. Please run data_collection.py first.")
    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()