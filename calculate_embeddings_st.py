;import pandas as pd
from sentence_transformers import SentenceTransformer
import time
from tqdm import tqdm

# Load the data
open_problems_articles_df = pd.read_csv("../data/final_articles.csv")

# Preprocessing: Combine article title and abstract into one field
preprocessed = open_problems_articles_df.dropna().copy()
preprocessed["article_text"] = preprocessed["article_title"] + " " + preprocessed["article_abstract"]

embeddings2_df = preprocessed.copy()

# Initialize the SentenceTransformer model
model = SentenceTransformer("neuml/pubmedbert-base-embeddings")

# Embedding function for queries and corpus
def get_embeddings(texts, model, batch_size=64):
    start_time = time.time()  # Record the start time
    
    # Compute the embeddings for the batch
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"Embedding complete. Elapsed time: {elapsed_time:.2f} seconds")
    
    return embeddings


open_problems_unique = embeddings2_df["open_problem_title"].unique()

print("Calculating embeddings for open problem titles...")
open_problem_embeddings = get_embeddings(open_problems_unique.tolist(), model, batch_size=64)

open_problem_embeddings_dict = dict(zip(open_problems_unique, open_problem_embeddings))

embeddings2_df['open_problem_embedding'] = embeddings2_df['open_problem_title'].map(open_problem_embeddings_dict)

article_texts_unique = embeddings2_df["article_text"].unique().tolist()

print("Calculating embeddings for article texts...")
article_text_embeddings = get_embeddings(article_texts_unique, model, batch_size=64)

article_text_embeddings_dict = dict(zip(article_texts_unique, article_text_embeddings))

embeddings2_df['article_text_embedding'] = embeddings2_df['article_text'].map(article_text_embeddings_dict)

output_path = "../data/all_embeddings2.csv"
embeddings2_df.to_csv(output_path, index=False, chunksize=5000)
print(f"Embeddings saved to {output_path}")
