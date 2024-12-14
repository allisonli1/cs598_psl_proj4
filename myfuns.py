import pandas as pd
import numpy as np

S = pd.read_csv('S_subset.csv')
MOVIE_LIST = S.columns.tolist()

MOVIE_INFO = pd.read_csv('https://liangfgithub.github.io/MovieData/movies.dat', sep='::', engine = 'python',
                         encoding="ISO-8859-1", header = None)
MOVIE_INFO.columns = ['movie_id', 'title', 'genres']

MOVIES_TOP10 = pd.read_csv('top10_movies.csv') 

def get_displayed_movies():
    return MOVIE_INFO.loc[MOVIE_INFO['movie_id'].isin([int(s.split('m')[-1]) for s in MOVIE_LIST])]


def myIBCF(w, S_3):
    movies = S_3.columns
    w = w.reshape(-1,)
    has_r_mask = ~np.isnan(w)

    new_ratings = np.zeros((len(movies),1))
    new_ratings[has_r_mask] = np.nan

    for idx in np.arange(len(movies)):
        if has_r_mask[idx]:
            continue
        s_i = S_3.iloc[idx,:].values.copy()
        s_i = np.nan_to_num(s_i, nan=0)

        if s_i[has_r_mask].sum() == 0: # NA
            predicted_rating = np.nan 
        else:
            predicted_rating = np.dot(w[has_r_mask],s_i[has_r_mask]) / s_i[has_r_mask].sum()

        new_ratings[idx] = predicted_rating
    
    predicted_R = pd.DataFrame({'rating': new_ratings.reshape(-1,)}).set_index(movies)
    top10_R = predicted_R.sort_values(by='rating', ascending=False).head(10)
    
    movie_recs = top10_R.dropna().index.tolist()
    print(movie_recs)
    if len(movie_recs) != 10: # there are NAs in top10
        print('HERE')
        num_na = 10 - len(movie_recs)
        movies_to_add = [m for m in MOVIES_TOP10['movie_id'] if m not in movie_recs]
        movie_recs.extend(movies_to_add[:num_na])

    return movie_recs

def get_recommended_movies(rating_input):
    print(rating_input)

    movie_list_int = [int(m.split('m')[-1]) for m in MOVIE_LIST]

    complete_ratings = [rating_input[m] if m in rating_input.keys() else np.nan for m in movie_list_int]
    complete_ratings = np.array(complete_ratings)

    print(complete_ratings)
    
    movie_rec_ids = myIBCF(complete_ratings, S)
    print(movie_rec_ids)
    movie_rec_ids = [int(m.split('m')[-1]) for m in movie_rec_ids]

    movie_recs = pd.DataFrame({'movie_id': movie_rec_ids})
    movie_recs = movie_recs.merge(MOVIE_INFO, how='left', on='movie_id')
    return movie_recs
