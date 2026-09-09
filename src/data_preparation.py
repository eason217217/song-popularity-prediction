# Dataset: "114k Spotify Songs" by Priyam Choksi (Kaggle)
# https://www.kaggle.com/datasets/priyamchoksi/spotify-dataset-114k-songs
# Download tracks.csv / artists.csv from the link above and place them in ../data/
import os
import pandas as pd


tracks_file_path = os.path.join(os.path.dirname(__file__), "..", "data", "tracks.csv")
artists_file_path = os.path.join(os.path.dirname(__file__), "..", "data", "artists.csv")

print("Tracks file path:", os.path.abspath(tracks_file_path))
print("Artists file path:", os.path.abspath(artists_file_path))


tracks = pd.read_csv(tracks_file_path)
artists = pd.read_csv(artists_file_path)

print(tracks.head())
print(artists.head())


tracks.dropna(inplace=True)


features = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]
target = 'popularity'


X = tracks[features]
y = tracks[target]


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


processed_data = pd.DataFrame(X_scaled, columns=features)
processed_data['popularity'] = y.values
processed_data.to_csv('processed_tracks.csv', index=False)
