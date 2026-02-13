import pandas as pd

# Load dataset
df = pd.read_csv('data/netflix_titles.csv')

# Select relevant columns and fill missing values
df = df[['title', 'type', 'director', 'cast', 'listed_in', 'description', 'release_year', 'rating', 'duration']]
df.fillna('', inplace=True)

# Create a 'tags' column by combining all text features
df['tags'] = df['description'] + " " + df['listed_in'] + " " + df['cast'] + " " + df['director'] + " " + df['type']

# Clean the tags (convert to lowercase)
df['tags'] = df['tags'].apply(lambda x: x.lower())

# Keep necessary columns for the app
final_df = df[['title', 'type', 'tags', 'listed_in', 'description', 'release_year', 'rating', 'duration']]

# Save the processed data
final_df.to_csv('data/processed_data.csv', index=False)
print("Preprocessing complete! 'processed_data.csv' created.")
print(f"Total records processed: {len(final_df)}")
print(f"Movies: {len(final_df[final_df['type'] == 'Movie'])}")
print(f"TV Shows: {len(final_df[final_df['type'] == 'TV Show'])}")