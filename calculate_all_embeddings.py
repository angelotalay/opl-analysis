import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# with open("../data/all_articles.xml", mode="r") as file: 
#     efetch_results = file.readlines() 

# def extract_information(text, dictionary): 
#     split_text = text.split("\t")[1:]
#     if len(split_text) == 2:# Create a Dataset class
# class ProblemArticleDataset(Dataset):
#     def __init__(self, dataframe):
#         self.dataframe = dataframe

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, idx):
#         row = self.dataframe.iloc[idx]
#         return row["open_problem_title"], row["article_text"]
#         dictionary["article_title"].append(split_text[0].strip("[]."))
#         dictionary["article_abstract"].append(split_text[1])

# all_articles_dict = {"article_title" : [], "article_abstract" : []}
# for entry in efetch_results:
#     extract_information(entry, all_articles_dict)

# # Creating DataFrame from dictionary
# all_articles_df = pd.DataFrame(all_articles_dict)
# all_articles_df["article_text"] = all_articles_df["article_title"] + ". " + all_articles_df["article_abstract"]

# Loading open problems data
open_problems = pd.read_csv("../data/selected-open-problems1.csv", skip_rows="1")
open_problem_embeddings = open_problems.loc[:, ["open_problem_title"]]

model = SentenceTransformer("neuml/pubmedbert-base-embeddings")
open_problem_embeddings_list = model.encode(open_problem_embeddings["open_problem_title"].tolist(), 
                                            batch_size=64, show_progress_bar=True)

open_problem_embeddings["embedding"] = open_problem_embeddings_list.tolist()

article_embeddings_list = model.encode(all_articles_df["article_text"].tolist(), 
                                       batch_size=64, show_progress_bar=True)

all_articles_df["embedding"] = article_embeddings_list.tolist()

open_problem_embeddings.to_parquet("../data/similarity_scores/open_problem_embeddings.parquet")
all_articles_df.to_parquet("../data/similarity_scores/article_embeddings.parquet")