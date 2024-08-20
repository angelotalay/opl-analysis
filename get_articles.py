import sh

import pandas as pd

import ast

searched_articles = pd.read_csv("../data/title_keywords.csv", skiprows=1)


def construct_query(keywords, use_all=True):
    """Constructs a query string from the given keywords and MeSH term."""
    ageing_query = "(Ageing[MeSH Terms])"
    if use_all and len(keywords) > 1:
        extracted_keywords = [f"({keyword[0]})" for keyword in keywords[:2] if
                              isinstance(keyword, tuple) and len(keyword) > 0]
        keyword_query = " AND ".join(extracted_keywords)
    else:
        if isinstance(keywords[0], tuple) and len(keywords[0]) > 0:
            keyword_query = f"({keywords[0][0]})"
        else:
            keyword_query = "(Ageing)"  # Fallback in case the keyword structure is incorrect

    return f"{keyword_query} AND {ageing_query}"


def perform_search(query, output_path):
    # Run the command using sh
    sh.esearch("-db", "pubmed", "-query", query, _piped=True) | \
    sh.efetch("-format", "xml", _piped=True) | \
    sh.xtract("-pattern", "PubmedArticle", "-element", "MedlineCitation/PMID,ArticleTitle,AbstractText",
              _out=output_path)


def search_pubmed(title, keywords, backup_keywords):
    """Searches PubMed with the given title and keywords, and uses backup keywords if necessary."""
    record_dict = {}

    # Use backup keywords if the initial keywords are empty
    if not keywords:
        keywords = backup_keywords

    keywords = ast.literal_eval(keywords)

    # Construct and perform the initial query with two keywords
    query = construct_query(keywords, use_all=True)
    record = perform_search(query, output_path="../data/pubmed_data")

    # Check if results are found, if not, retry with only the first keyword
    if record["Count"] == "0":
        query = construct_query(keywords, use_all=False)
        record = perform_search(query)

    # Store the results in a dictionary
    record_dict["counts"] = record["Count"]
    record_dict["ids"] = record["IdList"]
    record_dict["query"] = query

    return record_dict


first_ten_open_problems = searched_articles.head(10)

for index, row in first_ten_open_problems.iterrows():
    ...
