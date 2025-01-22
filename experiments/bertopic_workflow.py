import bertopic
import gc
import torch
from hdbscan import HDBSCAN
from umap import UMAP


def test_model(dataset, datasize, min_topic_size, preprocessed, iteration, seed, nr_topics, file_path, embedding_model,
               top_n_words=15):
    try:
        print(f"Testing: datasize={datasize}, min_topic_size={min_topic_size}")

        umap_model = UMAP(n_neighbors=15, n_components=5,
                          min_dist=0.0, metric='cosine', random_state=seed)

        topic_model = bertopic.BERTopic(nr_topics=nr_topics, top_n_words=top_n_words, min_topic_size=min_topic_size,
                                        umap_model=umap_model, calculate_probabilities=False)

        topic_model.fit_transform(dataset)

        print("Model done fitting")

        # to get keywords per topic, all we need to do is get_topics() on the topic model
        keywords_representation = {}
        for topic, value in topic_model.get_topics().items():

            # will give topic:keywords
            keywords = []
            for keyword, c_tf_idf in value:
                keywords.append(keyword)

            keywords_representation[topic] = keywords

        bertopic_labels = topic_model.generate_topic_labels(nr_words=3)

        with open(
                f"topics_keywords_datasize={datasize}_mintopicsize={min_topic_size}_nrwords15_preprocessed={preprocessed}_iteration{iteration}_seed{seed}_rerun.txt",
                "w") as file:
            for topic, keywords in keywords_representation.items():
                file.write(f"Topic {topic}: \n")
                file.write(f"Topic Keywords: {keywords} \n")
                file.write(
                    "----------------------------------------------------------------------------------------------------------------------------")
                file.write("\n")

        # BERTopic labels
        with open(
                f"bertopic_labels_datasize={datasize}_mintopicsize={min_topic_size}_nrwords15_preprocessed={preprocessed}_iteration{iteration}_seed{seed}_rerun.txt",
                "w") as file:
            for label in bertopic_labels:
                topic = int(label.split('_')[0])
                file.write(f"\nTopic {topic}: \n")
                file.write(f"Topic Keywords: {keywords_representation[topic]}\n")
                file.write(f"BERTopic Generated Label: {label} \n")
                file.write(
                    "----------------------------------------------------------------------------------------------------------------------------")
                file.write("\n")

        topic_model.save(file_path, serialization="safetensors", save_ctfidf=True,
                         save_embedding_model=embedding_model)

    except Exception as e:

        print(f"Exception occurred with these params: datasize={datasize}, min_topic_size={min_topic_size} \n\n")
        print(f"Exception: {e}")

    else:
        print(f"Successful run: datasize={datasize}, min_topic_size={min_topic_size} \n\n")

    finally:
        del topic_model
        gc.collect()


def visualize_topics(model_path):
    pass


def visualize_documents(model_path):
    pass


def get_word_weights(model_path):
    pass


if __name__ == "__main__":
    print(torch.cuda.is_available())
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    dataset_filenames = []
