import os
import pandas as pd


class TSDataset:
    def __init__(self):
        self.data_df = pd.DataFrame()


class NewselaDataset(TSDataset):
    def __init__(self, limit=None):
        super().__init__()
        self.data_dir = "./../../datasets/newsela/newsela_article_corpus_2016-01-29/articles"
        self.metadata_path = "./../../datasets/newsela/newsela_article_corpus_2016-01-29/articles_metadata.csv"
        self.limit = limit # None as default, else select int to slice
        # dataset metainfo
        self.dataset_name = "newsela"
        self.domain = "news"
        self.annotation = "human"
        self.alignment_level = "text"

        # relevant columns
        self.grade_level = "grade_level"
        self.simplification_version = "version"
        self.text = "content"
        self.language = "language"
        self.grouping_tag = "slug"

    def load_data(self):
        """ Loads Newsela onto a pandas df. """
        metadata = pd.read_csv(self.metadata_path, sep=",")
        data = []
        loaded_files = 0

        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.data_dir, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    data.append({"filename": filename, "content": text})
                    loaded_files += 1
                    if self.limit and loaded_files >= self.limit:
                        break

        articles_df = pd.DataFrame(data)
        self.data_df = pd.merge(articles_df, metadata, on="filename", how="inner")

    
class MedEASiDataset(TSDataset):
    def __init__(self, limit=None):
        super().__init__()
        self.file_path = "./../../datasets/Med-EASi/Med-EASi.full.ori.csv"
        self.limit = limit # None as default, else select int to slice
        # dataset metainfo
        self.dataset_name = "med_easi"
        self.domain = "medical"
        self.annotation = "human"
        self.alignment_level = "sentence"

        # relevant columns
        self.source_text = "Expert"
        self.target_text = "Simple"
        self.split = "split"
        self.grouping_tag = "idx"

    def load_data(self):
        """ Loads Med-EASi onto a pandas df. """
        try:
            data = pd.read_csv(self.file_path, sep=",")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

        if self.limit:
            data = data.head(self.limit)

        self.data_df = data


# dataset = MedEASiDataset(limit=10)
# dataset.load_data()
# print(dataset.data_df)


# # class WikiLargeDataset(TSDataset):
# #     def __init__(self, base_path, base_filename):
# #         super().__init__()
# #         self.base_path = base_path
# #         self.base_filename = base_filename

# #     def load_data(self):
# #         valid_src_path = os.path.join(self.base_path, f"{self.base_filename}valid.src")
# #         valid_dst_path = os.path.join(self.base_path, f"{self.base_filename}valid.dst")

# #         with open(valid_src_path, "r") as src_file, open(valid_dst_path, "r") as dst_file:
# #             valid_src = src_file.readlines()
# #             valid_dst = dst_file.readlines()

# #         valid_df = pd.DataFrame({"source": valid_src, "target": valid_dst})
# #         self.data_df = valid_df[valid_df["source"].str.strip().astype(bool) & valid_df["target"].str.strip().astype(bool)]



