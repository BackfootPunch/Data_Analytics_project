import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from gensim import corpora, models
from gensim.models import LdaModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

class LyricsNLPAnalyzer:
    def __init__(self, data_file: str = 'lyrics_corpus.csv'):
        """Initialize the NLP analyzer with lyrics data."""
        self.df = pd.read_csv(data_file)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add music-specific stop words
        music_stop_words = {
            'verse', 'chorus', 'bridge', 'outro', 'intro', 'yeah', 'oh', 'ah',
            'na', 'la', 'da', 'ooh', 'whoa', 'hey', 'yo', 'uh', 'mmm'
        }
        self.stop_words.update(music_stop_words)
        
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare and clean the lyrics data."""
        # Remove rows with missing lyrics
        self.df = self.df.dropna(subset=['lyrics'])
        
        # Clean lyrics
        self.df['lyrics_cleaned'] = self.df['lyrics'].apply(self.clean_lyrics)
        
        # Remove empty lyrics after cleaning
        self.df = self.df[self.df['lyrics_cleaned'].str.len() > 0]
        
        print(f"Prepared {len(self.df)} tracks with lyrics for analysis")
    
    def clean_lyrics(self, text: str) -> str:
        """Clean and preprocess lyrics text."""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove common patterns in lyrics
        patterns_to_remove = [
            r'\[.*?\]',  # Remove text in brackets like [Verse 1]
            r'\(.*?\)',  # Remove text in parentheses
            r'lyrics.*genius',  # Remove genius-specific text
            r'\d+embed',  # Remove embed numbers
            r'you might also like',  # Remove suggestions
            r'see.*live',  # Remove live performance references
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text, removing stop words."""
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove non-alphabetic tokens and stop words, then lemmatize
        processed_tokens = []
        for token in tokens:
            if (token.isalpha() and 
                len(token) > 2 and 
                token.lower() not in self.stop_words):
                lemmatized = self.lemmatizer.lemmatize(token.lower())
                processed_tokens.append(lemmatized)
        
        return processed_tokens
    
    def perform_sentiment_analysis(self) -> pd.DataFrame:
        """Perform sentiment analysis on lyrics."""
        sentiments = []
        
        for idx, row in self.df.iterrows():
            lyrics = row['lyrics_cleaned']
            blob = TextBlob(lyrics)
            
            sentiment_data = {
                'track_id': row['track_id'],
                'track_name': row['track_name'],
                'artist': row['artist'],
                'genre': row['genre'],
                'decade': row['decade'],
                'polarity': blob.sentiment.polarity,  # -1 to 1 (negative to positive)
                'subjectivity': blob.sentiment.subjectivity,  # 0 to 1 (objective to subjective)
                'word_count': len(lyrics.split()),
                'unique_words': len(set(lyrics.split()))
            }
            
            # Categorize sentiment
            if sentiment_data['polarity'] > 0.1:
                sentiment_data['sentiment_category'] = 'positive'
            elif sentiment_data['polarity'] < -0.1:
                sentiment_data['sentiment_category'] = 'negative'
            else:
                sentiment_data['sentiment_category'] = 'neutral'
            
            sentiments.append(sentiment_data)
        
        sentiment_df = pd.DataFrame(sentiments)
        return sentiment_df
    
    def perform_topic_modeling(self, num_topics: int = 10) -> Tuple[LdaModel, corpora.Dictionary]:
        """Perform LDA topic modeling on lyrics."""
        print("Tokenizing lyrics for topic modeling...")
        
        # Tokenize all lyrics
        tokenized_lyrics = []
        for lyrics in self.df['lyrics_cleaned']:
            tokens = self.tokenize_and_lemmatize(lyrics)
            if len(tokens) > 5:  # Only include lyrics with sufficient content
                tokenized_lyrics.append(tokens)
        
        if len(tokenized_lyrics) == 0:
            print("No suitable lyrics found for topic modeling")
            return None, None
        
        print(f"Processing {len(tokenized_lyrics)} documents for topic modeling...")
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(tokenized_lyrics)
        
        # Filter extremes
        dictionary.filter_extremes(no_below=5, no_above=0.7)
        
        # Create corpus
        corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_lyrics]
        
        # Train LDA model
        print(f"Training LDA model with {num_topics} topics...")
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        return lda_model, dictionary
    
    def extract_topics_by_decade(self, lda_model: LdaModel, dictionary: corpora.Dictionary) -> pd.DataFrame:
        """Extract dominant topics for each decade and genre."""
        topic_data = []
        
        for decade in self.df['decade'].unique():
            decade_data = self.df[self.df['decade'] == decade]
            
            for genre in decade_data['genre'].unique():
                genre_data = decade_data[decade_data['genre'] == genre]
                
                # Combine all lyrics for this genre-decade combination
                combined_lyrics = ' '.join(genre_data['lyrics_cleaned'].tolist())
                tokens = self.tokenize_and_lemmatize(combined_lyrics)
                
                if len(tokens) > 10:
                    # Get topic distribution
                    bow = dictionary.doc2bow(tokens)
                    topic_dist = lda_model.get_document_topics(bow)
                    
                    # Find dominant topic
                    if topic_dist:
                        dominant_topic = max(topic_dist, key=lambda x: x[1])
                        topic_id, topic_prob = dominant_topic
                        
                        # Get topic words
                        topic_words = lda_model.show_topic(topic_id, topn=10)
                        top_words = [word for word, prob in topic_words]
                        
                        topic_data.append({
                            'decade': decade,
                            'genre': genre,
                            'dominant_topic_id': topic_id,
                            'topic_probability': topic_prob,
                            'top_words': ', '.join(top_words[:5]),
                            'all_topic_words': ', '.join(top_words),
                            'track_count': len(genre_data)
                        })
        
        return pd.DataFrame(topic_data)
    
    def create_wordclouds_by_decade(self):
        """Create word clouds for each decade."""
        decades = self.df['decade'].unique()
        
        fig, axes = plt.subplots(1, len(decades), figsize=(15, 5))
        if len(decades) == 1:
            axes = [axes]
        
        for i, decade in enumerate(decades):
            decade_data = self.df[self.df['decade'] == decade]
            
            # Combine all lyrics for the decade
            combined_lyrics = ' '.join(decade_data['lyrics_cleaned'].tolist())
            
            # Tokenize and clean
            tokens = self.tokenize_and_lemmatize(combined_lyrics)
            text_for_wordcloud = ' '.join(tokens)
            
            if text_for_wordcloud:
                # Create word cloud
                wordcloud = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    max_words=100,
                    colormap='viridis'
                ).generate(text_for_wordcloud)
                
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f'{decade}', fontsize=14, fontweight='bold')
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'No data\nfor {decade}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{decade}', fontsize=14, fontweight='bold')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('wordclouds_by_decade.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Word clouds saved as wordclouds_by_decade.png")
    
    def analyze_sentiment_trends(self, sentiment_df: pd.DataFrame):
        """Analyze sentiment trends across decades and genres."""
        # Create sentiment trend visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Average sentiment by decade
        decade_sentiment = sentiment_df.groupby('decade').agg({
            'polarity': 'mean',
            'subjectivity': 'mean'
        }).reset_index()
        
        axes[0, 0].bar(decade_sentiment['decade'], decade_sentiment['polarity'])
        axes[0, 0].set_title('Average Sentiment Polarity by Decade')
        axes[0, 0].set_ylabel('Polarity (-1 to 1)')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # 2. Sentiment distribution
        sentiment_counts = sentiment_df['sentiment_category'].value_counts()
        axes[0, 1].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Overall Sentiment Distribution')
        
        # 3. Sentiment by genre
        genre_sentiment = sentiment_df.groupby(['genre', 'decade'])['polarity'].mean().unstack()
        genre_sentiment.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Average Sentiment by Genre and Decade')
        axes[1, 0].set_ylabel('Polarity')
        axes[1, 0].legend(title='Decade')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Subjectivity by decade
        axes[1, 1].bar(decade_sentiment['decade'], decade_sentiment['subjectivity'])
        axes[1, 1].set_title('Average Subjectivity by Decade')
        axes[1, 1].set_ylabel('Subjectivity (0 to 1)')
        
        plt.tight_layout()
        plt.savefig('sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Sentiment analysis plots saved as sentiment_analysis.png")
    
    def run_complete_analysis(self):
        """Run the complete NLP analysis pipeline."""
        print("Starting NLP analysis...")
        
        # 1. Sentiment Analysis
        print("Performing sentiment analysis...")
        sentiment_df = self.perform_sentiment_analysis()
        sentiment_df.to_csv('sentiment_analysis.csv', index=False)
        print(f"Sentiment analysis completed for {len(sentiment_df)} tracks")
        
        # 2. Topic Modeling
        print("Performing topic modeling...")
        lda_model, dictionary = self.perform_topic_modeling(num_topics=8)
        
        if lda_model and dictionary:
            # Extract topics by decade
            topics_df = self.extract_topics_by_decade(lda_model, dictionary)
            topics_df.to_csv('topics_by_decade.csv', index=False)
            
            # Save topic details
            topics_details = []
            for topic_id in range(lda_model.num_topics):
                topic_words = lda_model.show_topic(topic_id, topn=15)
                topics_details.append({
                    'topic_id': topic_id,
                    'keywords': ', '.join([word for word, prob in topic_words]),
                    'word_probabilities': ', '.join([f"{word}:{prob:.3f}" for word, prob in topic_words])
                })
            
            topics_details_df = pd.DataFrame(topics_details)
            topics_details_df.to_csv('topic_details.csv', index=False)
            
            print(f"Topic modeling completed with {lda_model.num_topics} topics")
        
        # 3. Create visualizations
        print("Creating word clouds...")
        self.create_wordclouds_by_decade()
        
        print("Creating sentiment trend analysis...")
        self.analyze_sentiment_trends(sentiment_df)
        
        # 4. Summary statistics
        self.create_summary_report(sentiment_df)
        
        print("NLP analysis completed!")
        print("Generated files:")
        print("- sentiment_analysis.csv")
        print("- topics_by_decade.csv")
        print("- topic_details.csv")
        print("- wordclouds_by_decade.png")
        print("- sentiment_analysis.png")
        print("- nlp_summary_report.txt")
    
    def create_summary_report(self, sentiment_df: pd.DataFrame):
        """Create a summary report of the NLP analysis."""
        report = []
        report.append("=== NLP ANALYSIS SUMMARY REPORT ===\n")
        
        # Overall statistics
        report.append(f"Total tracks analyzed: {len(sentiment_df)}")
        report.append(f"Average sentiment polarity: {sentiment_df['polarity'].mean():.3f}")
        report.append(f"Average subjectivity: {sentiment_df['subjectivity'].mean():.3f}")
        report.append("")
        
        # Sentiment by decade
        report.append("SENTIMENT BY DECADE:")
        decade_stats = sentiment_df.groupby('decade').agg({
            'polarity': ['mean', 'std'],
            'subjectivity': ['mean', 'std']
        }).round(3)
        
        for decade in sentiment_df['decade'].unique():
            decade_data = sentiment_df[sentiment_df['decade'] == decade]
            report.append(f"{decade}:")
            report.append(f"  Average polarity: {decade_data['polarity'].mean():.3f}")
            report.append(f"  Average subjectivity: {decade_data['subjectivity'].mean():.3f}")
            report.append(f"  Track count: {len(decade_data)}")
            
            # Most positive and negative tracks
            most_positive = decade_data.loc[decade_data['polarity'].idxmax()]
            most_negative = decade_data.loc[decade_data['polarity'].idxmin()]
            
            report.append(f"  Most positive: '{most_positive['track_name']}' by {most_positive['artist']} ({most_positive['polarity']:.3f})")
            report.append(f"  Most negative: '{most_negative['track_name']}' by {most_negative['artist']} ({most_negative['polarity']:.3f})")
            report.append("")
        
        # Sentiment by genre
        report.append("SENTIMENT BY GENRE:")
        for genre in sentiment_df['genre'].unique():
            genre_data = sentiment_df[sentiment_df['genre'] == genre]
            report.append(f"{genre}:")
            report.append(f"  Average polarity: {genre_data['polarity'].mean():.3f}")
            report.append(f"  Track count: {len(genre_data)}")
            report.append("")
        
        # Write report to file
        with open('nlp_summary_report.txt', 'w') as f:
            f.write('\n'.join(report))


def main():
    """Main function to run NLP analysis."""
    try:
        # Initialize analyzer
        analyzer = LyricsNLPAnalyzer()
        
        # Run complete analysis
        analyzer.run_complete_analysis()
        
    except FileNotFoundError:
        print("Error: lyrics_corpus.csv not found. Please run data_collection.py first.")
    except Exception as e:
        print(f"Error during NLP analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()