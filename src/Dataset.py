import os
import pandas as pd


class TSDataset:
    def __init__(self):
        self.data_df = pd.DataFrame()


class NewselaDataset(TSDataset):
    def __init__(self, data_dir, metadata_path, limit=None):
        super().__init__()
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.limit = limit
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

    def preprocess_text(self):
        pass


# # NEWSELA_DIR_PATH = "./../datasets/newsela/newsela_article_corpus_2016-01-29/articles"
# # NEWSELA_METADATA_PATH = "./../datasets/newsela/newsela_article_corpus_2016-01-29/articles_metadata.csv"

# # dataset = NewselaDataset(NEWSELA_DIR_PATH, NEWSELA_METADATA_PATH, limit=10)
# # dataset.load_data()
# # print(dataset.data_df.head())  # Check if preprocessing worked

# # df = dataset.data_df
# # print(df.columns)

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


# # class MedEASiDataset(TSDataset):
# #     def __init__(self, file_path):
# #         super().__init__()
# #         self.file_path = file_path

# #     def load_data(self):
# #         self.data_df = pd.read_csv(self.file_path)
# #         self.data_df.rename(columns={"Expert": "source", "Simple": "target"}, inplace=True)
# #         self.data_df = self.data_df[
# #             self.data_df["source"].str.strip().astype(bool) & self.data_df["target"].str.strip().astype(bool)
# #         ]
