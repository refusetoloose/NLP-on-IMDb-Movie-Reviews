# Natural Language Processing on IMDb Movie Reviews

This repository contains code and data for performing Natural Language Processing (NLP) on movie reviews extracted from IMDb. The tasks include text processing, topic modeling using Latent Dirichlet Allocation (LDA), and training custom word embeddings using Word2Vec models. The project leverages popular Python libraries such as Pandas, Seaborn, scikit-learn, Gensim, and spaCy.

## Project Structure

- `requirements.txt`: A file listing all the dependencies needed to run the code.
- `.gitignore`: A file specifying which files should be ignored by Git.
- `train.csv`: Preprocessed dataset of IMDb movie reviews.
- `Processing Movie Reviews.py`: Script for processing movie reviews using spaCy.
- `Topic Modeling on Movie Reviews.py`: Script for performing topic modeling on movie reviews using LDA.
- `Training Word Embedding on Movie Reviews.py`: Script for training custom word embeddings using Word2Vec.

## Setup Instructions

### Prerequisites

- Python (version 3.7 or higher)
- Visual Studio Code
- Git

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/refusetoloose/NLP-on-IMDb-Movie-Reviews.git
   NLP-on-IMDb-Movie-Reviews

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Download spaCy's English language model:**
   ```bash
   python -m spacy download en_core_web_sm

## Scripts and Tasks
1. **Processing Movie Reviews:**
   
   This script processes the movie reviews using spaCy. It performs the following tasks:
   - Counts the number of sentences in each movie review.
   - Extracts named entities of types PERSON, NORP, FAC, and ORG.
   - Extracts adjectives from the reviews.

   **Usage:**
   Run the Processing Movie Reviews.py script to generate insights and visualizations:
   -   python Processing Movie Reviews.py    
   
   **Generated Graphs:**
   -  Number of Sentences w.r.t Sentiment
   -  Top 20 Entities in Selected Types
   -  Top 20 Adjectives w.r.t Sentiment

3. **Topic Modeling on Movie Reviews:**
   
   This script performs topic modeling using the Latent Dirichlet Allocation (LDA) model. It uses spaCy for tokenization and scikit-learn's CountVectorizer and LatentDirichletAllocation.

   **Usage:**
   Run the Topic Modeling on Movie Reviews.py script to identify and visualize topics:
   -   python Topic Modeling on Movie Reviews.py

   **Sample Output:**
   
   Topic 1:  
   love wife father woman life mother young son man husband  
   ...  
   ...    
   Topic 19:  
   war people american world documentary history japanese political america anti  

5. **Training Word Embedding on Movie Reviews**
   
   This script applies pre-trained word embedding models and trains custom word embedding models using Gensim's Word2Vec.

   **Usage:**
   Run the Training Word Embedding on Movie Reviews.py script to train and evaluate word embeddings:
   -   python Training Word Embedding on Movie Reviews.py

   **Sample Output:**
     
   similarity between funny and comedy is 0.46052005887031555  
   similarity between funny and music is 0.22704412043094635  
   similarity between funny and laugh is 0.4182666540145874  
   similarity between funny and humor is 0.4602951407432556  
   similarity between funny and song is 0.14965760707855225  
   similarity between funny and songs is 0.31771841645240784  
   similarity between funny and jokes is 0.388815015554428  

   ## Contributing  
   Contributions are welcome! Please fork this repository and submit pull requests with your improvements.

   ## License 
   This project is licensed under the MIT License. See the LICENSE file for more details.

   ## Acknowledgements 
   - spaCy
   - Gensim
   - scikit-learn
   - Pandas
   - Seaborn
  
   ## Reference 
   Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word 
   vectors for sentiment analysis. The 49th Annual Meeting of the Association for 
   Computational Linguistics (ACL 2011).
