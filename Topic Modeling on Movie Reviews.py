#%% Prajwal Assignment 4: Lab 2 Topic Modelling Movie Reviews
# %%
import pandas as pd
import spacy
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")
data = pd.read_csv("train.csv")
print(data["review"].head())

#%% Use spaCy to process the text. 
def spacy_tokenizer(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

#%%
tf_vectorizer = CountVectorizer(
    # set up your CountVectorizer
    tokenizer=spacy_tokenizer, max_df=0.95, min_df=2, max_features=1000
)

#%%
with tqdm(total=len(data)) as pbar:
    tf = tf_vectorizer.fit_transform(data["review"])
    pbar.update(len(data))
    
#%%    
print(tf_vectorizer.get_feature_names_out()[:10])
#%%
lda = LatentDirichletAllocation(
    # set up your LatentDirichletAllocation
    n_components=20, learning_method='online', learning_offset=50, random_state=69
)
#%%
with tqdm(total=100) as pbar:
    lda.fit(tf)
    pbar.update(100)

#%%

def show_topic(model, feature_names, top):
    for index, distribution in enumerate(model.components_):
        sorted_word_indices = distribution.argsort()[::-1][:top]
        print(f"Topic {index}:")
        print(" ".join([feature_names[i] for i in sorted_word_indices]))

tf_feature_names = tf_vectorizer.get_feature_names_out()
top = 10
show_topic(lda, tf_feature_names, top)


# %%
