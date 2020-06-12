import pandas as pd
import numpy as np
import warnings
from constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)
datapath = 'D:/weiterbildung/recommender systems/movielens-20m-dataset/rating.csv'
separator = ","
_size = "20m"
DEFAULT_HEADER = (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)

# Warning and error messages
WARNING_MOVIE_LENS_HEADER = """MovieLens rating dataset has four columns
    (user id, movie id, rating, and timestamp), but more than four column names are provided.
    Will only use the first four column names."""
WARNING_HAVE_SCHEMA_AND_HEADER = """Both schema and header are provided.
    The header argument will be ignored."""
ERROR_HEADER = "Header error. At least user and movie column names should be provided"


def load_pandas_df(
        header=None,
        local_cache_path=None,
        title_col=None,
        genres_col=None,
        year_col=None,
):
    size = _size
    size = size.lower()

    if header is None:
        header = DEFAULT_HEADER
    elif len(header) < 2:
        raise ValueError(ERROR_HEADER)
    elif len(header) > 4:
        warnings.warn(WARNING_MOVIE_LENS_HEADER)
        header = header[:4]

    movie_col = header[1]

    # Load rating data
    df = pd.read_csv(
        datapath,
        sep=separator,
        engine="python",
        names=header,
        usecols=[*range(len(header))],
        header=0,
    )

    # Convert 'rating' type to float
    if len(header) > 2:
        df[header[2]] = df[header[2]].astype(float)

    df['rating'] = df['rating'].astype(np.float32)
    return df