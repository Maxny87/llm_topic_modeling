import os
import bertopic
import gc
import torch
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP
import pandas as pd

def bertopic_workflow(dataset, seed, min_topic_size, num_topics, csv_file_path,
         bertopic_labels_csv_file_path, save_model=False, model_file_path=''):
    try:
        print(f"Testing: dataset={dataset}, min_top_size={min_topic_size}, num_topics={num_topics}")


        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=seed)
        hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size, metric='euclidean', cluster_selection_method='eom',
                                prediction_data=True)

        topic_model = bertopic.BERTopic(nr_topics=num_topics, top_n_words=15, umap_model=umap_model, calculate_probabilities=False,
                                        hdbscan_model=hdbscan_model, embedding_model=embedding_model)

        topic_model.fit_transform(dataset)

        print("Model done fitting")

        keywords_representation = {}
        for topic, value in topic_model.get_topics().items():
            keywords = []
            for keyword, c_tf_idf in value:
                keywords.append(keyword)

            keywords_representation[topic] = keywords

        bertopic_labels = topic_model.generate_topic_labels(nr_words=3)

        # Topics with their 15 keywords
        with open(csv_file_path, "w", encoding='utf-8') as file:
            for topic, keywords in keywords_representation.items():
                file.write(f"Topic {topic}: \n")
                file.write(f"Topic Keywords: {keywords} \n")
                file.write("----------------------------------------------------------------------------------------------------------------------------")
                file.write("\n")

        # BERTopic labels
        with open(bertopic_labels_csv_file_path, "w", encoding='utf-8') as file:
            for label in bertopic_labels:
                topic = int(label.split('_')[0])
                file.write(f"\nTopic {topic}: \n")
                file.write(f"Topic Keywords: {keywords_representation[topic]}\n")
                file.write(f"BERTopic Generated Label: {label} \n")
                file.write("----------------------------------------------------------------------------------------------------------------------------")
                file.write("\n")

        if save_model:
            topic_model.save(model_file_path, serialization="safetensors", save_ctfidf=True,
                         save_embedding_model=embedding_model)

    except Exception as e:

        print(f"Exception occurred with these params: dataset={dataset}, min_top_size={min_topic_size}, num_topics={num_topics} \n\n")
        print(f"Exception: {e}")

    else:
        print(f"Successful run: dataset={dataset}, min_top_size={min_topic_size}, num_topics={num_topics} \n\n")

    finally:
        del topic_model, hdbscan_model, umap_model, dataset
        torch.cuda.empty_cache()
        gc.collect()

def plot_umap(topic_model, save_file_path, dataset_name):
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

def plot_hbdscan(topic_model, data, save_file_path, dataset_name):
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    sentence_model = SentenceTransformer(embedding_model)
    umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=1)

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

def plot_wordweights(topic_model, save_file_path, nr_topics):
    fig = topic_model.visualize_barchart(top_n_topics=nr_topics)
    fig.update_layout(
        title=dict(
            text="Newsgroup20: Top Words Per Topic",
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
    test_dict = {
        "arxiv_abstracts": {
            "data": pd.read_csv("clean_arxiv_abstracts_final.csv"),
            "min_topic_size": 300,
            "nr_topics": 26,
            "seed": 1,
            "model_save_file_path": "./models/arxiv_abstracts",
            "topic_labels_save_file_path": "./results/arxiv_abstracts.csv",
            "bertopic_labels_save_file_path": "./results/arxiv_abstracts_with_bertopic_labels.csv",
        },
        "amazon_reviews": {
            "data": pd.read_csv("clean_amazon_reviews_final.csv"),
            "min_topic_size": 400,
            "nr_topics": 21,
            "seed": 1,
            "model_save_file_path": "./models/amazon_reviews",
            "topic_labels_save_file_path": "./results/amazon_reviews.csv",
            "bertopic_labels_save_file_path": "./results/amazon_reviews_with_bertopic_labels.csv",
        },
        "worldcup2022": {
            "data": pd.read_csv("clean_worldcup_final.csv"),
            "min_topic_size": [250, 350],
            "nr_topics": [16, 26],
            "seed": 1,
            "model_save_file_path": ["./models/worldcup2022_mintopicsize250_nrtopics16", "./models/worldcup2022_mintopicsize350_nrtopics26"],
            "topic_labels_save_file_path": ["./results/worldcup2022_mintopicsize250_nrtopics16.csv", "./results/worldcup2022_mintopicsize350_nrtopics26.csv"],
            "bertopic_labels_save_file_path": ["./results/worldcup2022_mintopicsize250_nrtopics16_with_bertopic_labels.csv", "./results/worldcup2022_mintopicsize350_nrtopics26_with_bertopic_labels.csv"],
        },
        "bbc_news": {
            "data": pd.read_csv("clean_bbc_news_dataset_final.csv"),
            "min_topic_size": 15,
            "nr_topics": 16,
            "seed": 1,
            "model_save_file_path": "./models/bbc_news",
            "topic_labels_save_file_path": "./results/bbc_news.csv",
            "bertopic_labels_save_file_path": "./results/bbc_news_with_bertopic_labels.csv",
        },
        "newsgroup20": {
            "data": pd.read_csv("clean_newsgroup20_final.csv"),
            "min_topic_size": 25,
            "nr_topics": 26,
            "seed": 1,
            "model_save_file_path": "./models/newsgroup20",
            "topic_labels_save_file_path": "./results/newsgroup20.csv",
            "bertopic_labels_save_file_path": "./results/newsgroup20_with_bertopic_labels.csv",
        },
    }

    for dataset, params in test_dict.items():
        if dataset == "worldcup2022":
            for i in range(len(params["min_topic_size"])):
                bertopic_workflow(params["data"], params["seed"], params["min_topic_size"][i], params["nr_topics"][i], params["topic_labels_save_file_path"][i],
                    params["bertopic_labels_save_file_path"][i], save_model=True, model_file_path=params["model_save_file_path"][i])
        else:
            bertopic_workflow(params['data'], params['seed'], params['min_topic_size'], params['nr_topics'], params['topic_labels_save_file_path'],
                              params['bertopic_labels_save_file_path'], save_model=True, model_file_path=params['model_save_file_path'])

if __name__ == "__main__":
    test()
