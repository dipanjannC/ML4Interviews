import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.scaler = StandardScaler()
        self.item_profiles = None
        self.similarity_matrix = None
        self.items_df = None
        
    def create_sample_data(self):
        """Create sample movie dataset for demonstration"""
        movies_data = {
            'movie_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'title': [
                'The Dark Knight', 'Inception', 'Interstellar', 'The Godfather',
                'Pulp Fiction', 'The Shawshank Redemption', 'Forrest Gump',
                'The Matrix', 'Goodfellas', 'Fight Club'
            ],
            'genres': [
                'Action Crime Drama', 'Action Sci-Fi Thriller', 'Adventure Drama Sci-Fi',
                'Crime Drama', 'Crime Drama', 'Drama', 'Drama Romance',
                'Action Sci-Fi', 'Biography Crime Drama', 'Drama Thriller'
            ],
            'director': [
                'Christopher Nolan', 'Christopher Nolan', 'Christopher Nolan',
                'Francis Ford Coppola', 'Quentin Tarantino', 'Frank Darabont',
                'Robert Zemeckis', 'Lana Wachowski', 'Martin Scorsese', 'David Fincher'
            ],
            'cast': [
                'Christian Bale Heath Ledger', 'Leonardo DiCaprio Marion Cotillard',
                'Matthew McConaughey Anne Hathaway', 'Marlon Brando Al Pacino',
                'John Travolta Samuel L Jackson', 'Tim Robbins Morgan Freeman',
                'Tom Hanks Robin Wright', 'Keanu Reeves Laurence Fishburne',
                'Robert De Niro Ray Liotta', 'Brad Pitt Edward Norton'
            ],
            'description': [
                'Batman fights crime in Gotham City with help from allies',
                'A thief enters peoples dreams to steal secrets',
                'Astronauts travel through wormhole to save humanity',
                'Aging patriarch transfers control of crime family to son',
                'Hitman stories interweave in Los Angeles',
                'Banker wrongly imprisoned finds hope and friendship',
                'Man with low IQ experiences historical events',
                'Hacker discovers reality is computer simulation',
                'Rise and fall of mobster Henry Hill',
                'Insomniac office worker starts underground fight club'
            ],
            'rating': [9.0, 8.8, 8.6, 9.2, 8.9, 9.3, 8.8, 8.7, 8.7, 8.8],
            'year': [2008, 2010, 2014, 1972, 1994, 1994, 1994, 1999, 1990, 1999]
        }
        return pd.DataFrame(movies_data)
    
    def preprocess_features(self, df):
        """Combine textual features for content analysis"""
        df['combined_features'] = (
            df['genres'] + ' ' + 
            df['director'] + ' ' + 
            df['cast'] + ' ' + 
            df['description']
        )
        return df
    
    def fit(self, items_df):
        """Train the content-based recommender"""
        self.items_df = items_df.copy()
        
        # Preprocess features
        self.items_df = self.preprocess_features(self.items_df)
        
        # Create TF-IDF matrix for combined textual features
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.items_df['combined_features'])
        
        # Calculate cosine similarity matrix
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        
        print(f"Model trained on {len(self.items_df)} items")
        print(f"Feature matrix shape: {tfidf_matrix.shape}")
        
    def get_recommendations(self, item_id, n_recommendations=5):
        """Get recommendations based on item similarity"""
        if self.similarity_matrix is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Get index of the item
        try:
            item_idx = self.items_df[self.items_df['movie_id'] == item_id].index[0]
        except IndexError:
            raise ValueError(f"Item ID {item_id} not found in dataset")
        
        # Get similarity scores for all items
        sim_scores = list(enumerate(self.similarity_matrix[item_idx]))
        
        # Sort items by similarity (excluding the item itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get indices of most similar items (excluding the item itself)
        similar_items_indices = [i[0] for i in sim_scores[1:n_recommendations+1]]
        
        # Return recommended items
        recommendations = self.items_df.iloc[similar_items_indices][
            ['movie_id', 'title', 'genres', 'director', 'rating']
        ].copy()
        
        # Add similarity scores
        recommendations['similarity_score'] = [sim_scores[i+1][1] for i in range(n_recommendations)]
        
        return recommendations
    
    def get_item_profile(self, item_id):
        """Get the feature profile of an item"""
        item_data = self.items_df[self.items_df['movie_id'] == item_id]
        if item_data.empty:
            raise ValueError(f"Item ID {item_id} not found")
        
        return item_data.iloc[0]
    
    def plot_similarity_matrix(self, figsize=(10, 8)):
        """Visualize the similarity matrix"""
        plt.figure(figsize=figsize)
        
        # Create labels for the heatmap
        labels = [f"{row['title'][:15]}" for _, row in self.items_df.iterrows()]
        
        sns.heatmap(
            self.similarity_matrix, 
            xticklabels=labels, 
            yticklabels=labels,
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            square=True
        )
        plt.title('Movie Similarity Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def explain_recommendation(self, item_id, recommended_item_id):
        """Explain why an item was recommended"""
        base_item = self.get_item_profile(item_id)
        recommended_item = self.get_item_profile(recommended_item_id)
        
        print(f"\nWhy '{recommended_item['title']}' was recommended based on '{base_item['title']}':")
        print(f"Base item genres: {base_item['genres']}")
        print(f"Recommended item genres: {recommended_item['genres']}")
        print(f"Base item director: {base_item['director']}")
        print(f"Recommended item director: {recommended_item['director']}")
        
        # Calculate similarity score
        base_idx = self.items_df[self.items_df['movie_id'] == item_id].index[0]
        rec_idx = self.items_df[self.items_df['movie_id'] == recommended_item_id].index[0]
        similarity = self.similarity_matrix[base_idx][rec_idx]
        print(f"Similarity score: {similarity:.3f}")

def main():
    # Create and train the recommender
    recommender = ContentBasedRecommender()
    
    # Create sample data
    movies_df = recommender.create_sample_data()
    print("Sample Movies Dataset:")
    print(movies_df[['title', 'genres', 'director', 'rating']].head())
    
    # Train the model
    print("\n" + "="*50)
    print("Training Content-Based Recommender...")
    recommender.fit(movies_df)
    
    # Get recommendations for a specific movie
    print("\n" + "="*50)
    base_movie_id = 2  # Inception
    print(f"Getting recommendations based on: {movies_df[movies_df['movie_id'] == base_movie_id]['title'].iloc[0]}")
    
    recommendations = recommender.get_recommendations(base_movie_id, n_recommendations=3)
    print("\nRecommended Movies:")
    print(recommendations[['title', 'genres', 'director', 'rating', 'similarity_score']])
    
    # Explain a recommendation
    print("\n" + "="*50)
    recommended_movie_id = recommendations.iloc[0]['movie_id']
    recommender.explain_recommendation(base_movie_id, recommended_movie_id)
    
    # Show similarity matrix visualization
    print("\n" + "="*50)
    print("Generating similarity matrix visualization...")
    recommender.plot_similarity_matrix()
    
    # Demonstrate recommendations for different movies
    print("\n" + "="*50)
    print("Recommendations for different movies:")
    
    test_movies = [1, 4, 8]  # Dark Knight, Godfather, Matrix
    for movie_id in test_movies:
        movie_title = movies_df[movies_df['movie_id'] == movie_id]['title'].iloc[0]
        print(f"\nBased on '{movie_title}':")
        recs = recommender.get_recommendations(movie_id, n_recommendations=2)
        for _, rec in recs.iterrows():
            print(f"  - {rec['title']} (similarity: {rec['similarity_score']:.3f})")

if __name__ == "__main__":
    main()