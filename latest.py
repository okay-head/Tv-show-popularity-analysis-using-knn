
# ### **Imports**
# 
# 

# In[41]:


import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
import numpy as np
from matplotlib.pyplot import imread
import json
# import warnings
# warnings.filterwarnings('ignore')
# import base64
# import io
# import codecs
# from IPython.display import HTML


# ### **Dataset**

# In[42]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# ## **Data Exploration & Cleaning**

# In[43]:


movies.head()


# In[44]:


movies.describe()


# In[45]:


credits.head()


# In[46]:


credits.describe()


# **Converting JSON into strings**

# In[47]:


# changing the genres column from json to string
movies['genres'] = movies['genres'].apply(json.loads)
for index,i in zip(movies.index,movies['genres']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name'])) # the key 'name' contains the name of the genre
    movies.loc[index,'genres'] = str(list1)

# changing the keywords column from json to string
movies['keywords'] = movies['keywords'].apply(json.loads)
for index,i in zip(movies.index,movies['keywords']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    movies.loc[index,'keywords'] = str(list1)
    
# changing the production_companies column from json to string
movies['production_companies'] = movies['production_companies'].apply(json.loads)
for index,i in zip(movies.index,movies['production_companies']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    movies.loc[index,'production_companies'] = str(list1)

# changing the cast column from json to string
credits['cast'] = credits['cast'].apply(json.loads)
for index,i in zip(credits.index,credits['cast']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    credits.loc[index,'cast'] = str(list1)

# changing the crew column from json to string    
credits['crew'] = credits['crew'].apply(json.loads)
def director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
credits['crew'] = credits['crew'].apply(director)
credits.rename(columns={'crew':'director'},inplace=True)


# In[48]:


movies.head()


# In[49]:


movies.iloc[25]


# ### **Merging the two csv files**

# In[50]:


movies = movies.merge(credits,left_on='id',right_on='movie_id',how='left')
movies = movies[['id','original_title','genres','cast','vote_average','director','keywords']]


# In[51]:


movies.iloc[25]


# In[52]:


movies.shape


# In[53]:


movies.size


# In[54]:


movies.index


# In[55]:


movies.columns


# In[56]:


movies.dtypes


# ## **Working with the Genres column**

# In[57]:


movies['genres'] = movies['genres'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['genres'] = movies['genres'].str.split(',')


# In[58]:


plt.subplots(figsize=(12,10))
list1 = []
for i in movies['genres']:
    list1.extend(i)
ax = pd.Series(list1).value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('hls',10))
for i, v in enumerate(pd.Series(list1).value_counts()[:10].sort_values(ascending=True).values): 
    ax.text(.8, i, v,fontsize=12,color='white',weight='bold')
plt.title('Top Genres')
plt.show()


# Drama appears to be the most popular genre followed by Comedy.

# In[59]:


for i,j in zip(movies['genres'],movies.index):
    list2=[]
    list2=i
    list2.sort()
    movies.loc[j,'genres']=str(list2)
movies['genres'] = movies['genres'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['genres'] = movies['genres'].str.split(',')


# Now lets generate a list 'genreList' with all possible unique genres mentioned in the dataset.
# 
# 

# In[60]:


genreList = []
for index, row in movies.iterrows():
    genres = row["genres"]
    
    for genre in genres:
        if genre not in genreList:
            genreList.append(genre)
genreList[:10] #now we have a list with unique genres


# **One Hot Encoding for multiple labels**

# In[61]:


def binary(genre_list):
    binaryList = []
    
    for genre in genreList:
        if genre in genre_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList


# In[62]:


movies['genres_bin'] = movies['genres'].apply(lambda x: binary(x))
movies['genres_bin'].head()


# **Working with the Cast Column**
#  

# In[63]:


movies['cast'] = movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'').str.replace('"','')
movies['cast'] = movies['cast'].str.split(',')


# In[64]:


plt.subplots(figsize=(12,10))
list1=[]
for i in movies['cast']:
    list1.extend(i)
ax=pd.Series(list1).value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('muted',40))
for i, v in enumerate(pd.Series(list1).value_counts()[:15].sort_values(ascending=True).values): 
    ax.text(.8, i, v,fontsize=10,color='white',weight='bold')
plt.title('Actors with highest appearance')
plt.show()


# In[65]:


for i,j in zip(movies['cast'],movies.index):
    list2 = []
    list2 = i[:4]
    movies.loc[j,'cast'] = str(list2)
movies['cast'] = movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['cast'] = movies['cast'].str.split(',')
for i,j in zip(movies['cast'],movies.index):
    list2 = []
    list2 = i
    list2.sort()
    movies.loc[j,'cast'] = str(list2)
movies['cast']=movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'')


# In[66]:


castList = []
for index, row in movies.iterrows():
    cast = row["cast"]
    
    for i in cast:
        if i not in castList:
            castList.append(i)


# In[67]:


def binary(cast_list):
    binaryList = []
    
    for genre in castList:
        if genre in cast_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList


# In[68]:


movies['cast_bin'] = movies['cast'].apply(lambda x: binary(x))
movies['cast_bin'].head()


#  **Working with Director column**

# In[69]:


def xstr(s):
    if s is None:
        return ''
    return str(s)
movies['director'] = movies['director'].apply(xstr)


# In[70]:


plt.subplots(figsize=(12,10))
ax = movies[movies['director']!=''].director.value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('muted',40))
for i, v in enumerate(movies[movies['director']!=''].director.value_counts()[:10].sort_values(ascending=True).values): 
    ax.text(.5, i, v,fontsize=12,color='white',weight='bold')
plt.title('Directors with highest movies')
plt.show()


# In[71]:


directorList=[]
for i in movies['director']:
    if i not in directorList:
        directorList.append(i)


# In[72]:


def binary(director_list):
    binaryList = []  
    for direct in directorList:
        if direct in director_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    return binaryList


# In[73]:


movies['director_bin'] = movies['director'].apply(lambda x: binary(x))
movies.head()


# ## **Working with the Keywords column**

# In[76]:


from wordcloud import WordCloud, STOPWORDS
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords


# In[77]:


plt.subplots(figsize=(12,12))
stop_words = set(stopwords.words('english'))
stop_words.update(',',';','!','?','.','(',')','$','#','+',':','...',' ','')

words=movies['keywords'].dropna().apply(nltk.word_tokenize)
word=[]
for i in words:
    word.extend(i)
word=pd.Series(word)
word=([i for i in word.str.lower() if i not in stop_words])
wc = WordCloud(background_color="black", max_words=2000, stopwords=STOPWORDS, max_font_size= 60,width=1000,height=1000)
wc.generate(" ".join(word))
plt.imshow(wc)
plt.axis('off')
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.show()


# Above is a wordcloud showing the major keywords or tags used for describing the movies.
# 

# In[78]:


movies['keywords'] = movies['keywords'].str.strip('[]').str.replace(' ','').str.replace("'",'').str.replace('"','')
movies['keywords'] = movies['keywords'].str.split(',')
for i,j in zip(movies['keywords'],movies.index):
    list2 = []
    list2 = i
    movies.loc[j,'keywords'] = str(list2)
movies['keywords'] = movies['keywords'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['keywords'] = movies['keywords'].str.split(',')
for i,j in zip(movies['keywords'],movies.index):
    list2 = []
    list2 = i
    list2.sort()
    movies.loc[j,'keywords'] = str(list2)
movies['keywords'] = movies['keywords'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['keywords'] = movies['keywords'].str.split(',')


# In[79]:


words_list = []
for index, row in movies.iterrows():
    genres = row["keywords"]
    
    for genre in genres:
        if genre not in words_list:
            words_list.append(genre)


# In[80]:


def binary(words):
    binaryList = []
    for genre in words_list:
        if genre in words:
            binaryList.append(1)
        else:
            binaryList.append(0)
    return binaryList


# In[81]:


movies['words_bin'] = movies['keywords'].apply(lambda x: binary(x))
movies = movies[(movies['vote_average']!=0)] #removing the movies with 0 score and without drector names 
movies = movies[movies['director']!='']


# ## Similarity between movies

# We will we using Cosine Similarity for finding the similarity between 2 movies. 

# In[82]:


from scipy import spatial

def Similarity(movieId1, movieId2):
    a = movies.iloc[movieId1]
    b = movies.iloc[movieId2]
    
    genresA = a['genres_bin']
    genresB = b['genres_bin']
    
    genreDistance = spatial.distance.cosine(genresA, genresB)
    
    scoreA = a['cast_bin']
    scoreB = b['cast_bin']
    scoreDistance = spatial.distance.cosine(scoreA, scoreB)
    
    directA = a['director_bin']
    directB = b['director_bin']
    directDistance = spatial.distance.cosine(directA, directB)
    
    wordsA = a['words_bin']
    wordsB = b['words_bin']
    wordsDistance = spatial.distance.cosine(directA, directB)
    return genreDistance + directDistance + scoreDistance + wordsDistance


# In[83]:


Similarity(3,160) #checking similarity between any 2 random movies


# We see that the distance is about 2.068, which is high. The more the distance, the less similar the movies are. Let's see what these random movies actually were.

# In[84]:

# Commented out

#  print(movies.iloc[3])
#  print(movies.iloc[160])


# It is evident that The Dark Knight Rises and How to train your Dragon 2 are very different movies. Thus the distance is huge.
# 
# 

# In[85]:


new_id = list(range(0,movies.shape[0]))
movies['new_id']=new_id
movies=movies[['original_title','genres','vote_average','genres_bin','cast_bin','new_id','director','director_bin','words_bin']]
movies.head()


# ## **Score Predictor**

# In[86]:


import operator

def predict_score(name):
    #name = input('Enter a movie title: ')
    # changed this part
    filt = movies['original_title'].str.lower().str.contains(name)
    if(movies[filt].shape[0]==0):
        print('Oopsie not found!')
    else: 
        new_movie= movies[filt].iloc[0].to_frame().T
        print('Selected Movie: ',new_movie.original_title.values[0])
        
        def getNeighbors(baseMovie, K):
            distances = []
    
            for index, movie in movies.iterrows():
                if movie['new_id'] != baseMovie['new_id'].values[0]:
                    dist = Similarity(baseMovie['new_id'].values[0], movie['new_id'])
                    distances.append((movie['new_id'], dist))
        
            distances.sort(key=operator.itemgetter(1))
            neighbors = []
    
            for x in range(K):
                neighbors.append(distances[x])
            return neighbors
# -----------------------------
        K = 10
        avgRating = 0
        neighbors = getNeighbors(new_movie, K)
    
        print('\nRecommended Movies: \n')
        for neighbor in neighbors:
            avgRating = avgRating+movies.iloc[neighbor[0]][2]  
            print( movies.iloc[neighbor[0]][0]+" | Genres: "+str(movies.iloc[neighbor[0]][1]).strip('[]').replace(' ','')+" | Rating: "+str(movies.iloc[neighbor[0]][2]))
    
        print('\n')
        avgRating = avgRating/K
        print('The predicted rating for %s is: %f' %(new_movie['original_title'].values[0],avgRating))
        print('The actual rating for %s is %f' %(new_movie['original_title'].values[0],new_movie['vote_average']))
        predictArr.append(int(avgRating))
        actualArr.append(int(new_movie['vote_average']))
        
    #     print(predictArr,acutalArr,sep='\n')
    
while(1):
    inputName = input('\nEnter a movie title: ')
    
    predict_score(inputName)  

# predict_score('Godfather')


# In[88]:


# predict_score('Donnie Darko')


# In[89]:


# predict_score('Notting Hill')


# In[90]:


# predict_score('Despicable Me')


# In[91]:


#predict_score()

