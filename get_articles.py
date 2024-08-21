import glob
import re
import pandas as pd
import os
from tqdm import tqdm

FILE_PATH_PATTERN = "../data/pubmed_data/*.xml"
COLUMNS = ["open_problem_title", "pmid", "article_title", "article_abstract"]
OUTPUT_CSV_PATH = "../data/final_articles.csv"


def extract_articles(path: str) -> list:
    with open(path, mode="r") as file:
        articles = file.readlines()
    return articles


def match_to_open_problem(path: str, article_dataframe) -> str:
    # Regex pattern to extract digits
    pattern = r'op_(\d+)_articles\.xml'

    # Search for the pattern in the file path
    match = re.search(pattern, path)

    if match:
        # Extract the digits
        digits = match.group(1)
        index = int(digits) - 1
        open_problem = article_dataframe.iloc[index, :]
        title = open_problem["titles"]
        return title
    else:
        raise ValueError("File path does not match the expected pattern")


def get_article_data(title, article_array: list) -> pd.DataFrame:
    # Each item in the array is an article
    data = {
        "open_problem_title": [title] * len(article_array),
        "pmid": [],
        "article_title": [],
        "article_abstract": [],
    }
    for article in article_array:
        components = article.split("\t")
        pmid = components[0]
        article_title = components[1]
        data["pmid"].append(pmid)
        data["article_title"].append(article_title)
        abstract = components[2] if len(components) > 2 else ""
        data["article_abstract"].append(abstract)

    return pd.DataFrame(data)


def create_dataframe(columns: list) -> pd.DataFrame:
    dataframe = pd.DataFrame(columns=columns)
    return dataframe


def iterate(articles_dataframe, output_dataframe) -> pd.DataFrame:
    for file_path in tqdm(glob.glob(FILE_PATH_PATTERN)):
        content = extract_articles(file_path)
        open_problem_title = match_to_open_problem(file_path, articles_dataframe)
        data_df = get_article_data(open_problem_title, content)
        output_dataframe = pd.concat([output_dataframe, data_df], ignore_index=True)

    return output_dataframe


if __name__ == "__main__":
    searched_articles = pd.read_json("../data/article_ids.json")
    final_dataframe = create_dataframe(columns=COLUMNS)
    final_dataframe = iterate(searched_articles, final_dataframe)

    # Save the final dataframe to a CSV file
    final_dataframe.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Data successfully saved to {OUTPUT_CSV_PATH}")
