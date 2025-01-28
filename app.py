from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

app = Flask(__name__)
app.secret_key = 'your_secret_key'


# Load datasets
def load_data():
    global users, products, ratings, user_activity
    try:
        users = pd.read_csv("UserDataset.csv")
        products = pd.read_csv("market_product_dataset.csv")
        ratings = pd.read_csv("RatingsDataset.csv")
        user_activity = pd.read_csv("UserActivity.csv")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        users = pd.DataFrame(columns=["userId", "age", "gender", "location", "username", "password"])
        products = pd.DataFrame(columns=["productId", "title", "category", "price", "description"])
        ratings = pd.DataFrame(columns=["userId", "productId", "rating"])
        user_activity = pd.DataFrame(columns=["userId", "activityType", "timestamp", "productId"])


load_data()  # Initially load data


# Utility Functions
def create_user_item_matrix(ratings_df):
    user_mapper = {user: idx for idx, user in enumerate(ratings_df['userId'].unique())}
    product_mapper = {prod: idx for idx, prod in enumerate(ratings_df['productId'].unique())}

    user_index = ratings_df['userId'].map(user_mapper)
    product_index = ratings_df['productId'].map(product_mapper)

    user_item_matrix = csr_matrix((ratings_df['rating'], (user_index, product_index)),
                                  shape=(len(user_mapper), len(product_mapper)))
    return user_item_matrix, user_mapper, {v: k for k, v in product_mapper.items()}


def recommend_products(user_id=None, k=10):
    if products.empty or ratings.empty:
        return products.head(k)['title'].tolist()  # Return generic products if no data

    user_item_matrix, user_mapper, reverse_product_mapper = create_user_item_matrix(ratings)
    try:
        user_idx = user_mapper[user_id]
    except KeyError:
        user_idx = None

    # Collaborative Filtering
    if user_idx is not None:
        knn = NearestNeighbors(metric='cosine', algorithm='brute')
        knn.fit(user_item_matrix)
        user_vector = user_item_matrix[user_idx, :].toarray().flatten()
        distances, indices = knn.kneighbors([user_vector], n_neighbors=k + 10)
        recommended_ids_cf = [reverse_product_mapper[idx] for idx in indices.flatten() if idx in reverse_product_mapper]
    else:
        recommended_ids_cf = []

    # Content-Based Filtering
    tfidf = TfidfVectorizer(stop_words='english')
    products['description'] = products['description'].fillna('')
    tfidf_matrix = tfidf.fit_transform(products['description'])

    content_scores = tfidf_matrix.mean(axis=0).A1  # Average across products
    recommended_ids_cb = products.iloc[content_scores.argsort()[::-1]]['productId'].tolist()

    # Merge and filter recommendations
    final_recommendations = list(dict.fromkeys(recommended_ids_cb + recommended_ids_cf))[:k]
    return products[products['productId'].isin(final_recommendations)]['title'].tolist()


# Routes
@app.route('/')
def home():
    load_data()  # Reload data to ensure it's up to date

    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']

    # Get user's past search queries from UserActivity.csv
    past_search_queries = user_activity[user_activity['userId'] == user_id]['search_query'].tolist()

    # If no past search queries, recommend based on generic content-based filtering
    if not past_search_queries:
        user_recommendations = recommend_products(user_id=user_id, k=5)
    else:
        # Filter products based on the past search queries (Content-Based Filtering)
        search_query_filter = "|".join(past_search_queries)
        filtered_products = products[
            products['title'].str.contains(search_query_filter, case=False, na=False) |
            products['description'].str.contains(search_query_filter, case=False, na=False)
            ]
        user_recommendations = filtered_products['title'].head(10).tolist()

    return render_template('home.html', products=user_recommendations)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        global users
        # Generate a new user ID
        new_user_id = users['userId'].max() + 1 if not users.empty else 1

        # Create a new DataFrame for the new user
        new_user = pd.DataFrame([{'userId': new_user_id, 'username': username, 'password': password}])

        # Append the new user to the existing users DataFrame

        users = pd.concat([users, new_user], ignore_index=True)

        # Save the updated users DataFrame to the CSV file
        users.to_csv("UserDataset.csv", index=False)

        return redirect(url_for('login'))

    return render_template('register.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users[(users['username'] == username) & (users['password'] == password)]
        if not user.empty:
            session['user_id'] = int(user.iloc[0]['userId'])
            return redirect(url_for('home'))
    return render_template('login.html', error="Invalid credentials")


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/search', methods=['GET'])
def search():
    search_query = request.args.get('search_query', '').lower()  # Get the search query from the request
    user_id = session.get('user_id', 'guest')

    load_data()  # Reload data to ensure it's up to date

    if search_query:
        activity_data = pd.DataFrame([{
            "userId": int(user_id),
            "search_query": search_query,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }])

        # Append search activity to the CSV
        try:
            user_activity = pd.read_csv("UserActivity.csv")
            user_activity = pd.concat([user_activity, activity_data], ignore_index=True)  # Concatenate the new data
        except FileNotFoundError:
            user_activity = activity_data  # Create a new DataFrame if the file doesn't exist

        user_activity.to_csv("UserActivity.csv", index=False)

        # Get updated recommendations after search
        user_recommendations = recommend_products(user_id=user_id, k=10)

        # Filter products based on whether the query matches name, category, or description
        filtered_products = products[
            products['title'].str.contains(search_query, case=False, na=False) |
            products['category'].str.contains(search_query, case=False, na=False) |
            products['description'].str.contains(search_query, case=False, na=False)
            ]
    else:
        filtered_products = []  # If no query, no search results
        user_recommendations = []  # No recommendations if no query

    return render_template('home.html', search_results=filtered_products.to_dict(orient='records'),
                           products=user_recommendations)


if __name__ == '__main__':
    app.run(debug=True)
