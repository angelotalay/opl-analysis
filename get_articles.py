import sh

import pandas as pd

searched_articles = pd.read_csv("../data/selected-open-problems1.csv", skiprows=1)
print(searched_articles.head(10))
def fetch_articles(query, output_path):
    # Run the command using sh
    sh.esearch("-db", "pubmed", "-query", query, _piped=True) | \
    sh.efetch("-format", "xml", _piped=True) | \
    sh.xtract("-pattern", "PubmedArticle", "-element", "MedlineCitation/PMID,ArticleTitle,AbstractText", _out=output_path)
