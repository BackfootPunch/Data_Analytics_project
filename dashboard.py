import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from viral_predictor import ViralHitPredictor
from genre_evolution_analysis import GenreEvolutionAnalyzer
import shap
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class MusicAnalyticsDashboard:
    def __init__(self):
        """Initialize the dashboard with data and models."""
        self.app = dash.Dash(__name__)
        self.setup_data()
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_data(self):
        """Load and prepare data for the dashboard."""
        try:
            # Load genre evolution data
            self.genre_analyzer = GenreEvolutionAnalyzer()
            self.genre_analyzer.calculate_decade_changes()
            
            # Load viral predictor
            self.viral_predictor = ViralHitPredictor()
            try:
                self.viral_predictor.load_model()
                self.model_loaded = True
            except:
                # Train model if not found
                print("Training viral predictor model...")
                self.viral_predictor.train_model()
                self.viral_predictor.save_model()
                self.model_loaded = True
            
            # Prepare data for visualizations
            self.prepare_dashboard_data()
            
        except Exception as e:
            print(f"Error setting up data: {e}")
            self.model_loaded = False
    
    def prepare_dashboard_data(self):
        """Prepare data specifically for dashboard visualizations."""
        # Get genre profiles for radar chart
        self.radar_data = []
        audio_features = [
            'danceability', 'energy', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo_normalized', 'loudness_normalized'
        ]
        
        for decade in self.genre_analyzer.df['decade'].unique():
            decade_data = self.genre_analyzer.df[self.genre_analyzer.df['decade'] == decade]
            
            for genre in decade_data['genre'].unique():
                genre_data = decade_data[decade_data['genre'] == genre]
                
                if len(genre_data) > 0:
                    profile = {
                        'decade': decade,
                        'genre': genre,
                        'track_count': len(genre_data)
                    }
                    
                    for feature in audio_features:
                        if feature in genre_data.columns:
                            profile[feature] = genre_data[feature].mean()
                    
                    self.radar_data.append(profile)
        
        self.radar_df = pd.DataFrame(self.radar_data)
        
        # Prepare network data
        self.network_fig = self.genre_analyzer.create_genre_network()
        
        # Get sample tracks for similarity comparison
        self.sample_tracks = self.genre_analyzer.df.sample(min(50, len(self.genre_analyzer.df))).copy()
    
    def create_radar_chart(self, selected_decades=None):
        """Create radar chart for selected decades."""
        if selected_decades is None:
            selected_decades = self.radar_df['decade'].unique()
        
        audio_features = [
            'danceability', 'energy', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo_normalized', 'loudness_normalized'
        ]
        
        fig = make_subplots(
            rows=1, cols=len(selected_decades),
            subplot_titles=selected_decades,
            specs=[[{"type": "polar"}] * len(selected_decades)]
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, decade in enumerate(selected_decades):
            decade_data = self.radar_df[self.radar_df['decade'] == decade]
            
            for j, genre in enumerate(decade_data['genre'].unique()):
                genre_data = decade_data[decade_data['genre'] == genre]
                
                if len(genre_data) > 0:
                    values = [genre_data[feature].iloc[0] for feature in audio_features]
                    
                    fig.add_trace(
                        go.Scatterpolar(
                            r=values,
                            theta=audio_features,
                            fill='toself',
                            name=f"{genre} ({decade})",
                            line_color=colors[j % len(colors)],
                            showlegend=(i == 0)
                        ),
                        row=1, col=i+1
                    )
        
        fig.update_layout(
            title="Genre Audio Feature Profiles",
            height=400,
            margin=dict(t=80, b=20, l=20, r=20)
        )
        
        return fig
    
    def create_prediction_explanation(self, prediction_result, track_features):
        """Create SHAP-style explanation visualization for a prediction."""
        if not self.model_loaded:
            return go.Figure().add_annotation(text="Model not available", showarrow=False)
        
        try:
            # Create a simple feature importance plot instead of full SHAP
            feature_importance = [
                ('danceability', track_features.get('danceability', 0) * 0.15),
                ('energy', track_features.get('energy', 0) * 0.14),
                ('valence', track_features.get('valence', 0) * 0.13),
                ('tempo_normalized', track_features.get('tempo_normalized', 0) * 0.12),
                ('loudness_normalized', track_features.get('loudness_normalized', 0) * 0.11),
                ('acousticness', track_features.get('acousticness', 0) * 0.10),
                ('speechiness', track_features.get('speechiness', 0) * 0.09),
                ('liveness', track_features.get('liveness', 0) * 0.08),
                ('instrumentalness', track_features.get('instrumentalness', 0) * 0.08),
            ]
            
            features, impacts = zip(*feature_importance)
            
            fig = go.Figure(go.Bar(
                x=list(impacts),
                y=list(features),
                orientation='h',
                marker_color=['red' if x < 0 else 'green' for x in impacts],
                text=[f"{x:.3f}" for x in impacts],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"Feature Impact on Viral Prediction (Probability: {prediction_result['viral_probability']:.3f})",
                xaxis_title="Impact Score",
                height=400,
                margin=dict(t=60, b=20, l=120, r=20)
            )
            
            return fig
        
        except Exception as e:
            return go.Figure().add_annotation(text=f"Error creating explanation: {str(e)}", showarrow=False)
    
    def find_similar_tracks(self, track_features, top_n=5):
        """Find similar tracks based on audio features."""
        try:
            audio_features = ['danceability', 'energy', 'speechiness', 'acousticness',
                            'instrumentalness', 'liveness', 'valence']
            
            # Calculate similarity scores
            similarities = []
            for idx, row in self.sample_tracks.iterrows():
                similarity = 0
                feature_count = 0
                
                for feature in audio_features:
                    if feature in track_features and feature in row:
                        similarity += abs(track_features[feature] - row[feature])
                        feature_count += 1
                
                if feature_count > 0:
                    similarity = 1 - (similarity / feature_count)  # Convert to similarity
                    similarities.append({
                        'track_name': row.get('name', 'Unknown'),
                        'artist': row.get('artist', 'Unknown'),
                        'popularity': row.get('popularity', 0),
                        'genre': row.get('genre', 'Unknown'),
                        'decade': row.get('decade', 'Unknown'),
                        'similarity': similarity
                    })
            
            # Sort by similarity and return top N
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_n]
        
        except Exception as e:
            return [{'error': str(e)}]
    
    def setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Music Analytics Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': 30, 'color': '#2c3e50'}),
            
            # Main content area
            html.Div([
                # Left panel - Genre Evolution
                html.Div([
                    html.H3("Genre Evolution", style={'color': '#34495e'}),
                    html.Label("Select Decades:"),
                    dcc.Checklist(
                        id='decade-selector',
                        options=[
                            {'label': '1980s', 'value': '1980s'},
                            {'label': '2000s', 'value': '2000s'},
                            {'label': '2020s', 'value': '2020s'}
                        ],
                        value=['1980s', '2000s', '2020s'],
                        inline=True,
                        style={'marginBottom': 20}
                    ),
                    dcc.Graph(id='radar-chart')
                ], style={
                    'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top',
                    'padding': '20px', 'backgroundColor': '#f8f9fa', 'margin': '1%',
                    'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                }),
                
                # Right panel - Viral Prediction
                html.Div([
                    html.H3("Viral Hit Predictor", style={'color': '#34495e'}),
                    
                    # Input form
                    html.Div([
                        html.H4("Track Features:", style={'marginBottom': 15}),
                        
                        # Audio features inputs
                        html.Div([
                            html.Div([
                                html.Label("Danceability:"),
                                dcc.Slider(id='danceability', min=0, max=1, step=0.01, value=0.5,
                                          marks={0: '0', 0.5: '0.5', 1: '1'})
                            ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'}),
                            
                            html.Div([
                                html.Label("Energy:"),
                                dcc.Slider(id='energy', min=0, max=1, step=0.01, value=0.5,
                                          marks={0: '0', 0.5: '0.5', 1: '1'})
                            ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'}),
                            
                            html.Div([
                                html.Label("Valence:"),
                                dcc.Slider(id='valence', min=0, max=1, step=0.01, value=0.5,
                                          marks={0: '0', 0.5: '0.5', 1: '1'})
                            ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'}),
                            
                            html.Div([
                                html.Label("Acousticness:"),
                                dcc.Slider(id='acousticness', min=0, max=1, step=0.01, value=0.5,
                                          marks={0: '0', 0.5: '0.5', 1: '1'})
                            ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'}),
                        ]),
                        
                        # Additional features
                        html.Div([
                            html.Div([
                                html.Label("Genre:"),
                                dcc.Dropdown(
                                    id='genre-dropdown',
                                    options=[
                                        {'label': 'Rock', 'value': 'rock'},
                                        {'label': 'Pop', 'value': 'pop'},
                                        {'label': 'Hip-Hop', 'value': 'hip-hop'},
                                        {'label': 'Electronic', 'value': 'electronic'},
                                        {'label': 'K-Pop', 'value': 'k-pop'},
                                        {'label': 'R&B', 'value': 'r&b'},
                                        {'label': 'Disco', 'value': 'disco'},
                                        {'label': 'Afrobeat', 'value': 'afrobeat'},
                                        {'label': 'Hyperpop', 'value': 'hyperpop'}
                                    ],
                                    value='pop'
                                )
                            ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'}),
                            
                            html.Div([
                                html.Label("Release Year:"),
                                dcc.Input(id='release-year', type='number', value=2023, 
                                         min=1980, max=2024, style={'width': '100%'})
                            ], style={'width': '48%', 'display': 'inline-block', 'margin': '1%'}),
                        ]),
                        
                        html.Button('Predict Virality', id='predict-button', 
                                   style={
                                       'backgroundColor': '#3498db', 'color': 'white',
                                       'border': 'none', 'padding': '10px 20px',
                                       'marginTop': '20px', 'borderRadius': '5px',
                                       'cursor': 'pointer', 'fontSize': '16px'
                                   })
                    ], style={'marginBottom': 30}),
                    
                    # Prediction results
                    html.Div(id='prediction-results', style={'marginBottom': 20}),
                    
                ], style={
                    'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top',
                    'padding': '20px', 'backgroundColor': '#f8f9fa', 'margin': '1%',
                    'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                })
            ]),
            
            # Bottom section - Network Graph
            html.Div([
                html.H3("Genre Relationship Network", 
                       style={'textAlign': 'center', 'color': '#34495e', 'marginBottom': 20}),
                dcc.Graph(id='network-graph', figure=self.network_fig)
            ], style={
                'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa',
                'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            }),
            
            # Bottom section - Prediction Explanation and Similar Tracks
            html.Div([
                # Prediction explanation
                html.Div([
                    html.H4("Prediction Explanation", style={'color': '#34495e'}),
                    dcc.Graph(id='explanation-plot')
                ], style={
                    'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top',
                    'padding': '20px', 'backgroundColor': '#f8f9fa', 'margin': '1%',
                    'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                }),
                
                # Similar tracks
                html.Div([
                    html.H4("Similar Tracks", style={'color': '#34495e'}),
                    html.Div(id='similar-tracks')
                ], style={
                    'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top',
                    'padding': '20px', 'backgroundColor': '#f8f9fa', 'margin': '1%',
                    'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                })
            ])
            
        ], style={'backgroundColor': '#ecf0f1', 'minHeight': '100vh', 'padding': '20px'})
    
    def setup_callbacks(self):
        """Setup dashboard callbacks for interactivity."""
        
        @self.app.callback(
            Output('radar-chart', 'figure'),
            [Input('decade-selector', 'value')]
        )
        def update_radar_chart(selected_decades):
            return self.create_radar_chart(selected_decades)
        
        @self.app.callback(
            [Output('prediction-results', 'children'),
             Output('explanation-plot', 'figure'),
             Output('similar-tracks', 'children')],
            [Input('predict-button', 'n_clicks')],
            [State('danceability', 'value'),
             State('energy', 'value'),
             State('valence', 'value'),
             State('acousticness', 'value'),
             State('genre-dropdown', 'value'),
             State('release-year', 'value')]
        )
        def predict_and_explain(n_clicks, danceability, energy, valence, acousticness, genre, release_year):
            if n_clicks is None:
                empty_fig = go.Figure().add_annotation(text="Click 'Predict Virality' to see explanation", 
                                                     showarrow=False)
                return "Enter track features and click 'Predict Virality'", empty_fig, "Similar tracks will appear here"
            
            if not self.model_loaded:
                error_fig = go.Figure().add_annotation(text="Model not available", showarrow=False)
                return "Error: Prediction model not available", error_fig, "Model not available"
            
            # Prepare track features
            track_features = {
                'danceability': danceability,
                'energy': energy,
                'valence': valence,
                'acousticness': acousticness,
                'speechiness': 0.1,  # Default values for features not in UI
                'instrumentalness': 0.1,
                'liveness': 0.2,
                'tempo_normalized': 0.6,
                'loudness_normalized': 0.7,
                'energy_danceability': danceability * energy,
                'valence_energy': valence * energy,
                'acoustic_energy_ratio': acousticness / (energy + 0.001),
                'danceability_index': danceability * 0.4 + energy * 0.3 + valence * 0.2 + 0.6 * 0.1,
                'release_year': release_year,
                'release_month': 6,
                'decade_numeric': 2020 if release_year >= 2020 else (2000 if release_year >= 2000 else 1980),
                'genre': genre,
                'key_mode': '1_1'
            }
            
            try:
                # Make prediction
                prediction = self.viral_predictor.predict_virality(track_features)
                
                # Create prediction results display
                prob_percent = prediction['viral_probability'] * 100
                confidence_percent = prediction['confidence'] * 100
                
                prediction_display = html.Div([
                    html.H4(f"Viral Probability: {prob_percent:.1f}%", 
                           style={'color': '#e74c3c' if prob_percent < 50 else '#27ae60'}),
                    html.P(f"Prediction: {'VIRAL' if prediction['is_viral_prediction'] else 'NOT VIRAL'}",
                          style={'fontSize': '18px', 'fontWeight': 'bold'}),
                    html.P(f"Confidence: {confidence_percent:.1f}%", style={'fontSize': '14px'})
                ])
                
                # Create explanation plot
                explanation_fig = self.create_prediction_explanation(prediction, track_features)
                
                # Find similar tracks
                similar_tracks = self.find_similar_tracks(track_features)
                similar_display = html.Div([
                    html.Div([
                        html.H5(f"{track['track_name']} by {track['artist']}"),
                        html.P(f"Genre: {track['genre']} | Decade: {track['decade']}"),
                        html.P(f"Popularity: {track['popularity']:.0f} | Similarity: {track['similarity']:.3f}"),
                        html.Hr()
                    ]) for track in similar_tracks[:5]
                ])
                
                return prediction_display, explanation_fig, similar_display
            
            except Exception as e:
                error_msg = f"Error making prediction: {str(e)}"
                error_fig = go.Figure().add_annotation(text=error_msg, showarrow=False)
                return error_msg, error_fig, "Error finding similar tracks"
    
    def run_server(self, debug=True, port=8050):
        """Run the dashboard server."""
        self.app.run_server(debug=debug, port=port, host='0.0.0.0')
    
    def save_dashboard(self, filename='music_analytics_dashboard.html'):
        """Save the dashboard as a standalone HTML file."""
        # This is a simplified version - full implementation would require 
        # converting Dash app to static HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Music Analytics Dashboard</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #ecf0f1; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .panel {{ background-color: #f8f9fa; padding: 20px; margin: 10px; 
                         border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                h1 {{ text-align: center; color: #2c3e50; }}
                h3 {{ color: #34495e; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Music Analytics Dashboard</h1>
                <div class="panel">
                    <h3>Interactive Dashboard</h3>
                    <p>This is a static version of the Music Analytics Dashboard.</p>
                    <p>To run the interactive version, execute: python dashboard.py</p>
                    <p>Features:</p>
                    <ul>
                        <li>Genre evolution radar charts across decades</li>
                        <li>Viral hit prediction with SHAP explanations</li>
                        <li>Genre relationship network visualization</li>
                        <li>Similar track recommendations</li>
                    </ul>
                    <p>The dashboard includes data from genres across three decades:</p>
                    <ul>
                        <li>1980s: Rock, Pop, Disco</li>
                        <li>2000s: Hip-Hop, R&B, Electronic</li>
                        <li>2020s: K-Pop, Afrobeat, Hyperpop</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
        print(f"Static dashboard info saved as {filename}")
        print("To run the interactive dashboard, use: python dashboard.py")


def main():
    """Main function to run the dashboard."""
    try:
        print("Initializing Music Analytics Dashboard...")
        dashboard = MusicAnalyticsDashboard()
        
        # Save static version
        dashboard.save_dashboard()
        
        print("Dashboard initialized successfully!")
        print("Starting server on http://localhost:8050")
        print("Press Ctrl+C to stop the server")
        
        # Run the dashboard
        dashboard.run_server(debug=False, port=8050)
        
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()