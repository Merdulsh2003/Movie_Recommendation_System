# What are Recommendation System
# Types of Recommendation System
''' 1. Content Based (Recommand on the basis of content similarities)
    2. Collabrative Based Filttering (recommand on the basis of similarity of 2 users)
    3. Hybrid (Combination of both content and collabrative)'''
    
# in Project we design a Content Based recommendation System

#importing modules
import numpy as np
import pandas as pd
import ast #convert string to list
from ast import literal_eval
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer
import pickle


#importing csv files.
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

#merging both database on the basis of title

movies = pd.merge(movies,credits,on='title')
#Removing Unusefull Columns from the database
'''use full tags
    1.genre
    2.id
    3.keyword
    4.title
    5.overview
    6.cast
    7. crew'''
    
movies = movies[['id','title','overview','genres','keywords','cast','crew']]
#Designing a new database which consists of 3 columns
# Movie_id,title, Tags(merging overview,genre,keywords,cast,crew)

# checking if there is any null data exists or not
movies.isnull().sum()
#removing 3 movies whose data is not given
movies.dropna(inplace=True)

#checking if there is any duplicate data or not
movies.duplicated().sum()

def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

'''extracting top 3 actors name form each movie from column- cast'''

def convert3(obj):
    L=[]
    counter= 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L
    
movies['cast'] = movies['cast'].apply(convert3)

def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L
    
movies['crew'] = movies['crew'].apply(fetch_director)

'''column name overview is in string data type, so we convert it into list so that we easily concaginate with other columns'''

movies['overview'] = movies['overview'].apply(lambda x:x.split())

'''as you can see in genres, cast, crew and keywords there is space between two words, for ex- movie recommendation
we need to remove the space between movie and recommendation and write it as movierecommendation'''

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x ])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x ])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x ])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x ])

'''conceginate all the above 4 columns and making a new column Tags'''

movies['tags']= movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df= movies[['id','title','tags']]

#convert tags column from datatype list to string
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))

#convert tags column in lowercase, an error occur but ignore it
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
 
 
'''------------TEXT VECTORIZATION--------------'''
'''we need to find similarities btw two or more movies using the their respective tags. to achieve this 
we convert text(each tags) in the form of vector. There are various techniques used to convert text to vector
but we use BAG OF WORDS'''

cv = CountVectorizer(max_features=5000,stop_words='english')
vectors= cv.fit_transform(new_df['tags']).toarray()

'''staming is a technique by using which we can convert same meaning words into a single word. for example- played, playing, play == play'''

ps = PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags']=new_df['tags'].apply(stem)

similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_index = new_df[new_df['title']== movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
        
pickle.dump(new_df,open('movie.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))
    
pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))

print(new_df['id'])






