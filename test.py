import movielens20MLocal as dl
import movielens as dl1
from python_splitters import python_stratified_split
import sar_singlenode as SAR
from timer import Timer
from mf_keras import Mf_keras
from simple_splitter import split
MOVIELENS_DATA_SIZE = "1m"

data = dl1.load_pandas_df(MOVIELENS_DATA_SIZE)
print(data.head())
N = data.userID.max() + 1  # number of users
M = data.itemID.max() + 1  # number of movies

"""train, test = python_stratified_split(data, ratio=0.80, col_user='userID', col_item='itemID', seed=42)"""
train, test = split(data)
"""model = SAR(
    col_user="userID",
    col_item="itemID",
    col_rating="rating",
    col_timestamp="timestamp",
    similarity_type="jaccard",
    time_decay_coefficient=30,
    timedecay_formula=True
)"""

model = Mf_keras(N, M)
with Timer() as train_time:
    model.fit(train, test)
print("Took {} seconds for training.".format(train_time.interval))

with Timer() as test_time:
    top_k = model.recommend_k_items(test, remove_seen=True)
print("Took {} seconds for prediction.".format(test_time.interval))

print(top_k.head())