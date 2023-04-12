import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import average_precision_score
from sklearn.metrics import ndcg_score
from collections import defaultdict
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load data from csv files
books = pd.read_csv('Books.csv', encoding='ISO-8859-1', on_bad_lines='skip')
ratings = pd.read_csv('Ratings.csv', encoding='ISO-8859-1')
users = pd.read_csv('Users.csv', encoding='ISO-8859-1')

reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)

trainset, testset = train_test_split(data, test_size=.25)

#SVD Algorithm is used to train the model
algo = SVD()
algo.fit(trainset)

predictions = algo.test(testset)

pred_df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])

threshold = 8

pred_df['above_threshold'] = (pred_df['rui'] >= threshold).astype(int)
pred_df['predicted_above_threshold'] = (pred_df['est'] >= threshold).astype(int)

#Top 5 recommendations for 10 random users
top_n = defaultdict(list)
for uid, iid, true_r, est, _ in predictions:
    top_n[uid].append((iid, est,))

# Randomly select 10 users
random_users = np.random.choice(list(top_n.keys()), 10, replace=False)

for uid in random_users:
    user_ratings = top_n[uid]
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    # Get top 5 books
    top_books = [iid for (iid, _) in user_ratings[:5]]
    book_titles = books[books['ISBN'].isin(top_books)]['Book-Title'].values.tolist()
    print(f"User {uid} recommended books: {top_books}")
    print(f"Titles: {book_titles}\n")

precision, recall, f1score, support = precision_recall_fscore_support(
    pred_df['above_threshold'], pred_df['predicted_above_threshold'], average='binary'
)

unique_books_test = len(set(pred_df['iid']))
unique_books_above_threshold = len(set(pred_df.loc[pred_df['above_threshold'] == 1, 'iid']))
coverage = unique_books_above_threshold / unique_books_test

pred_df.sort_values(['uid', 'est'], ascending=[True, False], inplace=True)
map_score = np.mean([average_precision_score(pred_df.loc[pred_df['uid'] == uid, 'above_threshold'],
                                              pred_df.loc[pred_df['uid'] == uid, 'est'])
                     for uid in pred_df['uid'].unique()])

ndcg_score = ndcg_score(pred_df['above_threshold'].to_numpy().reshape(1, -1),
                        pred_df['est'].to_numpy().reshape(1, -1),
                        k=len(pred_df['iid'].unique()))

# Print performance metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1score)
print("Coverage:", coverage)
print("Mean Average Position (MAP):", map_score)
print("Normalized Discounted Cumulative Gain (NDCG):", ndcg_score)