# Data

This project uses the **114k Spotify Songs** dataset by Priyam Choksi, published on Kaggle:

https://www.kaggle.com/datasets/priyamchoksi/spotify-dataset-114k-songs

The raw dataset (`tracks.csv`, `artists.csv`) is not checked into this repository because
of its size and license terms. To reproduce the pipeline:

1. Download the dataset from the Kaggle link above.
2. Place `tracks.csv` (and `artists.csv`, if used) in this `data/` folder.
3. Run the scripts in `src/` in order: `data_preparation.py` → `feature_engineering.py` → `train_regressor.py`.
