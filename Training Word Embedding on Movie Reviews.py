#%% Prajwal Assignment 4: Lab 3 Training Word Embedding on Movie Reviews
# %%
from itertools import combinations

import spacy

nlp = spacy.load("en_core_web_sm")
text = (
    "funny comedy music laugh humor song songs jokes musical hilarious"
)
doc = nlp(text)

for token1, token2 in combinations(doc, 2):
    print(
        f"similarity between {token1} and {token2} is {token1.similarity(token2)}"
    )


# %%
import pandas as pd
from gensim.models import Word2Vec
from tqdm.autonotebook import tqdm

def preprocessText(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.pos_ in ('NOUN', 'VERB', 'ADJ', 'ADV')]
    return tokens

data = pd.read_csv("train.csv")
#%%
data['processedReview'] = data['review'].apply(preprocessText)
#%%
# train a Word2vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# %%
for token1, token2 in combinations(text.split(), 2):
    print(
        f"similarity between {token1} and {token2} is {model.wv.similarity(token1, token2)}"
    )

# %%
