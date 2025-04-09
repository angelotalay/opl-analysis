import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

open_problems_articles_df = pd.read_csv("../data/final_articles.csv")

# Preprocessing
preprocessed = open_problems_articles_df.dropna().copy()
preprocessed["article_text"] = preprocessed["article_title"] + " " + preprocessed["article_abstract"]

model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the batch embedding function
def get_embeddings(texts, batch_size=64):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.extend(batch_embeddings)
    return np.array(all_embeddings)

embeddings1_df = preprocessed.copy()

# Calculate embeddings for unique open problem titles and map them
open_problems_unique = embeddings1_df["open_problem_title"].unique()
open_problem_embeddings = get_embeddings(open_problems_unique.tolist(), batch_size=64)
open_problem_embeddings_dict = dict(zip(open_problems_unique, open_problem_embeddings))
embeddings1_df['open_problem_embedding'] = embeddings1_df['open_problem_title'].map(open_problem_embeddings_dict)

# Calculate embeddings for unique article texts in batches
article_texts_dict = {}
article_texts_unique = embeddings1_df["article_text"].unique().tolist()

for i in tqdm(range(0, len(article_texts_unique), 64)):
    batch_texts = article_texts_unique[i:i + 64]
    batch_embeddings = get_embeddings(batch_texts, batch_size=64)
    for text, embedding in zip(batch_texts, batch_embeddings):
        article_texts_dict[text] = embedding

embeddings1_df["article_text_embedding"] = embeddings1_df["article_text"].map(article_texts_dict)

# Save the embeddings DataFrame to a CSV file
embeddings1_df.to_csv("../data/all_embeddings2.csv", index=False)
