import pickle
import streamlit as st
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
import base64


st.header('Book Recommender System')

user_ids = pickle.load(open('user_ids_list.pkl','rb'))
ratings_full_df = pickle.load(open('ratings_full_df.pkl','rb'))
books_df = pickle.load(open('books_df.pkl','rb'))

recommended_books = pd.DataFrame()

ratings_train_df, ratings_test_df = train_test_split(ratings_full_df,
                                   stratify=ratings_full_df['User-ID'],
                                   test_size=0.20,
                                   random_state=42)


#Creating a sparse pivot table with users in rows and items in columns
users_items_pivot_matrix_df = ratings_train_df.pivot(index='User-ID',
                                                          columns='ISBN',
                                                          values='Book-Rating').fillna(0)

users_items_pivot_matrix = users_items_pivot_matrix_df.values
users_ids_list = list(users_items_pivot_matrix_df.index)


# The number of factors to factor the user-item matrix.
NUMBER_OF_FACTORS_MF = 15

#Performs matrix factorization of the original user item matrix
U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)

sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = users_items_pivot_matrix_df.columns, index=users_ids_list).transpose()


class CFRecommender:

    MODEL_NAME = 'Collaborative Filtering'

    def __init__(self, cf_predictions_df):
        self.cf_predictions_df = cf_predictions_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False).reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating content that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['ISBN'].isin(items_to_ignore)].sort_values('recStrength', ascending = False).head(topn)
        recommendations_df=recommendations_df.merge(books_df,on='ISBN',how='inner')
        recommendations_df=recommendations_df[['ISBN','Book-Title','Image-URL-M','recStrength']]
        recommended_books = recommendations_df.head(5)
      
        return recommendations_df


cf_recommender_model = CFRecommender(cf_preds_df)

#Indexing by personId to speed up the searches during evaluation
ratings_full_indexed_df = ratings_full_df.set_index('User-ID')
ratings_train_indexed_df = ratings_train_df.set_index('User-ID')
ratings_test_indexed_df = ratings_test_df.set_index('User-ID')

def get_items_interacted(UserID, interactions_df):
    interacted_items = interactions_df.loc[UserID]['ISBN']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

class ModelRecommender:

    # Function for getting the set of items which a user has not interacted with
    def get_not_interacted_items_sample(self, UserID, sample_size, seed=42):
        interacted_items = get_items_interacted(UserID, ratings_full_indexed_df)
        all_items = set(explicit_rating['ISBN'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(list(non_interacted_items), sample_size)
        return set(non_interacted_items_sample)

    # Function to verify whether a particular item_id was present in the set of top N recommended items
    def _verify_hit_top_n(self, item_id, recommended_items, topn):
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

    # Function to evaluate the performance of model for each user
    def evaluate_model_for_user(self, model, person_id):

        # Getting the items in test set
        interacted_values_testset = ratings_test_indexed_df.loc[person_id]

        if type(interacted_values_testset['ISBN']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['ISBN'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['ISBN'])])

        interacted_items_count_testset = len(person_interacted_items_testset)
        
        # Getting a ranked recommendation list from the model for a given user
        person_recs_df = model.recommend_items(person_id, items_to_ignore=get_items_interacted(person_id, ratings_train_indexed_df),topn=10000000000)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
          st.text(person_recs_df.iloc[0]['Book-Title'])
          st.image(person_recs_df.iloc[0]['Image-URL-M'])
       
        with col2:
          st.text(person_recs_df.iloc[1]['Book-Title'])
          st.image(person_recs_df.iloc[1]['Image-URL-M'])

        with col3:
          st.text(person_recs_df.iloc[2]['Book-Title'])
          st.image(person_recs_df.iloc[2]['Image-URL-M'])
      
        with col4:
          st.text(person_recs_df.iloc[3]['Book-Title'])
          st.image(person_recs_df.iloc[3]['Image-URL-M'])
     
        with col5:
          st.text(person_recs_df.iloc[4]['Book-Title'])
          st.image(person_recs_df.iloc[4]['Image-URL-M'])
  
        # Function to evaluate the performance of model at overall level
    def recommend_book(self, model ,userid):

        person_metrics = self.evaluate_model_for_user(model, userid)
        return

model_recommender = ModelRecommender()


selected_user = st.selectbox(
    " Select a User-Id from the dropdown",
    user_ids
)

if st.button('Show Recommendations'):
    model_recommender.recommend_book(cf_recommender_model,selected_user)


    

   
   

    
        
        
