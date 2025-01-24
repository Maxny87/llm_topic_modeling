import os
import bertopic
import gc
import torch
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP


def main():
    pass


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
        del topic_model
        del hdbscan_model
        del umap_model
        del dataset
        gc.collect()

if __name__ == "__main__":
    main()
