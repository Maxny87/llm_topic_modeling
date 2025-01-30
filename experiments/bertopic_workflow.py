import os
import bertopic
import gc
import torch
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP
import pandas as pd

def bertopic_workflow(dataset_name, dataset, seed, min_topic_size, num_topics, csv_file_path,
         bertopic_labels_csv_file_path, save_model=False, model_file_path=''):
    """
    Runs the bertopic workflow from: "Empowering Topic Modeling with Large Language Models (LLMs): A Comparative Study on Labeling Efficiency and Accuracy."

    params:
        dataset_name: name of the dataset for this run
        dataset: actual data as list of text
        seed: seed for umap
        min_topic_size: min number of documents per topic
        num_topics: number of topics to extract from bertopic
        csv_file_path: where to save topics and keywords
        bertopic_labels_csv_file_path: where to save topics and keywords with the BERTopic bag of words label
        save_model: whether you want to save the trained model or not
        model_file_path: where to save the trained model if so

    """

    print(f"Testing: dataset={dataset_name}, min_top_size={min_topic_size}, num_topics={num_topics}")

    # initializing the umap and hbdscan for BERTopic
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=seed)

    hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size, metric='euclidean', cluster_selection_method='eom',
                            prediction_data=True)

    topic_model = bertopic.BERTopic(nr_topics=num_topics, top_n_words=15, umap_model=umap_model,
                                    calculate_probabilities=False,
                                    hdbscan_model=hdbscan_model, embedding_model=embedding_model)

    topic_model.fit_transform(dataset) # fitting the model to the dataset

    # for each topic extract the keywords and save in dict
    keywords_representation = {}
    for topic, value in topic_model.get_topics().items():
        keywords = []
        for keyword, c_tf_idf in value:
            keywords.append(keyword)

        keywords_representation[topic] = keywords

    bertopic_labels = topic_model.generate_topic_labels(nr_words=3)

    # saving the results of topic modeling
    with open(csv_file_path, "w", encoding='utf-8') as file:
        for topic, keywords in keywords_representation.items():
            file.write(f"Topic {topic}: \n")
            file.write(f"Topic Keywords: {keywords} \n")
            file.write("----------------------------------------------------------------------------------------------------------------------------")
            file.write("\n")

    with open(bertopic_labels_csv_file_path, "w", encoding='utf-8') as file:
        for label in bertopic_labels:
            topic = int(label.split('_')[0])
            file.write(f"\nTopic {topic}: \n")
            file.write(f"Topic Keywords: {keywords_representation[topic]}\n")
            file.write(f"BERTopic Generated Label: {label} \n")
            file.write("----------------------------------------------------------------------------------------------------------------------------")
            file.write("\n")

    if save_model: # saving the model
        topic_model.save(model_file_path, serialization="safetensors", save_ctfidf=True,
                         save_embedding_model=embedding_model)

    del topic_model, hdbscan_model, umap_model, dataset
    torch.cuda.empty_cache()
    gc.collect()

    print("success")

def plot_umap(topic_model_path, save_file_path, dataset_name):
    """
    This function is used to plot the results of the UMAP model with the passed BERTopic to see the dataset after dimensionality reduction and to see the data reduced with their given topic
    """
    topic_model = bertopic.BERTopic.load(topic_model_path)
    fig = topic_model.visualize_topics()
    fig.update_layout(
        title=dict(
            text=f"{dataset_name}: Topic Distribution",
            font=dict(size=28, family="Arial", color="black", weight="bold"),
            x=0.5,
            y=0.98
        ),
        width=1200,
        height=900,
        margin=dict(l=50, r=50, t=90, b=50),
        font=dict(size=20),
    )
    fig.write_image(f"{save_file_path}.png")

def plot_hbdscan(topic_model_path, data, save_file_path, dataset_name):
    """
    This function is used to plot the hbdscan cluster results with the passed BERTopic to see the distribution of topics
    """
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    sentence_model = SentenceTransformer(embedding_model)
    umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=1)

    topic_model = bertopic.BERTopic.load(topic_model_path)

    embeddings = sentence_model.encode(data)
    reduced_embeddings = umap_model.fit_transform(embeddings)

    fig = topic_model.visualize_documents(
        data,
        reduced_embeddings=reduced_embeddings,
        hide_document_hover=True,
        hide_annotations=True
    )
    fig.update_layout(
        title=dict(
            text=f"{dataset_name}: Topic Distribution",
            font=dict(size=28, family="Arial", color="black", weight="bold"),
            x=0.5,
            y=0.98
        ),
        width=1200,
        height=900,
        margin=dict(l=50, r=50, t=90, b=50),
        font=dict(size=20),
        legend=dict(font=dict(size=20)),
    )
    fig.write_image(f"{save_file_path}.png")

def plot_wordweights(topic_model_path, save_file_path, dataset_name, nr_topics):
    """
    This function is used to plot the wordweights for the specified number of topics for a passed BERTopic model to see the top words per topic
    """
    topic_model = bertopic.BERTopic.load(topic_model_path)
    fig = topic_model.visualize_barchart(top_n_topics=nr_topics)
    fig.update_layout(
        title=dict(
            text=f"{dataset_name}: Top Words Per Topic",
            font=dict(size=28, family="Arial", color="black", weight="bold"),
            x=0.5,
            y=0.98
        ),
        width=1500,
        height=1000,
        margin=dict(l=50, r=50, t=90, b=50),
        font=dict(size=20),
    )

    # iterating over all x-axes in subplots
    for i in range(1, nr_topics+1):
        fig['layout'][f'xaxis{i}']['tickfont']['size'] = 16
        fig['layout'][f'xaxis{i}']['title']['font']['size'] = 18

    # annotation font sizes
    for annotation in fig['layout']['annotations']:
        annotation['font']['size'] = 16

    fig.write_image(f"{save_file_path}.png")

def test():
    """
    This function is used to run and test the bertopic workflow with 5 datasets. The worldcup dataset requires 2 runs
    """
    test_dict = {
        "arxiv_abstracts": {
            "data": pd.read_csv("../preprocessed_data/clean_arxiv_abstracts.csv")['text'].to_list(),
            "min_topic_size": 300,
            "nr_topics": 26,
            "seed": 1,
            "model_save_file_path": "arxiv_abstracts_model",
            "topic_labels_save_file_path": "arxiv_abstracts.csv",
            "bertopic_labels_save_file_path": "arxiv_abstracts_with_bertopic_labels.csv",
        },
        "amazon_reviews": {
            "data": pd.read_csv("../preprocessed_data/clean_amazon.csv")['text'].to_list(),
            "min_topic_size": 400,
            "nr_topics": 21,
            "seed": 1,
            "model_save_file_path": "amazon_reviews_model",
            "topic_labels_save_file_path": "amazon_reviews.csv",
            "bertopic_labels_save_file_path": "amazon_reviews_with_bertopic_labels.csv",
        },
        "worldcup2022": {
            "data": pd.read_csv("../preprocessed_data/clean_worldcup.csv")['text'].to_list(),
            "min_topic_size": [250, 350],
            "nr_topics": [16, 26],
            "seed": 1,
            "model_save_file_path": ["worldcup2022_mintopicsize250_nrtopics16_model", "worldcup2022_mintopicsize350_nrtopics26_model"],
            "topic_labels_save_file_path": ["worldcup2022_mintopicsize250_nrtopics16.csv", "worldcup2022_mintopicsize350_nrtopics26.csv"],
            "bertopic_labels_save_file_path": ["worldcup2022_mintopicsize250_nrtopics16_with_bertopic_labels.csv", "worldcup2022_mintopicsize350_nrtopics26_with_bertopic_labels.csv"],
        },
        "bbc_news": {
            "data": pd.read_csv("../preprocessed_data/clean_bbc_news.csv")['text'].to_list(),
            "min_topic_size": 15,
            "nr_topics": 16,
            "seed": 1,
            "model_save_file_path": "bbc_news_model",
            "topic_labels_save_file_path": "bbc_news.csv",
            "bertopic_labels_save_file_path": "bbc_news_with_bertopic_labels.csv",
        },
        "newsgroup20": {
            "data": pd.read_csv("../preprocessed_data/clean_newsgroup20.csv")['text'].to_list(),
            "min_topic_size": 25,
            "nr_topics": 26,
            "seed": 1,
            "model_save_file_path": "newsgroup20_model",
            "topic_labels_save_file_path": "newsgroup20.csv",
            "bertopic_labels_save_file_path": "newsgroup20_with_bertopic_labels.csv",
        },
    }

    for dataset, params in test_dict.items():
        if dataset == "worldcup2022":
            for i in range(len(params["min_topic_size"])):
                bertopic_workflow(dataset, params["data"], params["seed"], params["min_topic_size"][i], params["nr_topics"][i], params["topic_labels_save_file_path"][i],
                    params["bertopic_labels_save_file_path"][i], save_model=True, model_file_path=params["model_save_file_path"][i])
        else:
            bertopic_workflow(dataset, params['data'], params['seed'], params['min_topic_size'], params['nr_topics'], params['topic_labels_save_file_path'],
                              params['bertopic_labels_save_file_path'], save_model=True, model_file_path=params['model_save_file_path'])

if __name__ == "__main__":
    test()
