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
        self.dataset_name = "medeasi"
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


class WikiLargeDataset(TSDataset):
    def __init__(self, limit=None):
        super().__init__()
        self.data_dir = "./../../datasets/wiki/wikilarge"
        self.base_filename = "wiki.full.aner."
        self.limit = limit

        # Dataset metainfo
        self.dataset_name = "wikilarge"
        self.domain = "general"
        self.annotation = "automatic"
        self.alignment_level = "sentence"

        # Relevant columns
        self.source_text = "source"
        self.target_text = "target"
        self.split = "split"
        self.grouping_tag = "idx"

    def load_data(self):

        splits = ["train", "valid", "test"]
        all_data = []

        for split in splits:
            src_path = os.path.join(self.data_dir, f"{self.base_filename}{split}.src")
            dst_path = os.path.join(self.data_dir, f"{self.base_filename}{split}.dst")

            # check if the files exist
            if not os.path.exists(src_path) or not os.path.exists(dst_path):
                print(f"Warning: Missing files for {split} split.")
                continue

            with open(src_path, "r", encoding="utf-8") as src_file, open(dst_path, "r", encoding="utf-8") as dst_file:
                src_lines = src_file.readlines()
                dst_lines = dst_file.readlines()

            # Ensure same number of lines in source and target
            if len(src_lines) != len(dst_lines):
                print(f"Warning: Mismatched line counts in {split} split.")

            # to df with cols source, target and split
            split_df = pd.DataFrame({
                "source": src_lines,
                "target": dst_lines,
                "split": split
            })

            # for dev
            if self.limit:
                split_df = split_df.head(self.limit)

            all_data.append(split_df)

        # combine into one df
        if all_data:
            self.data_df = pd.concat(all_data, ignore_index=True)
            self.data_df['idx'] = pd.Series(range(1, len(self.data_df) + 1), index=self.data_df.index)
        else:
            print("Error: No data loaded.")


# dataset = WikiLargeDataset(limit=10)
# dataset.load_data()
# print(dataset.data_df)


# dataset = MedEASiDataset(limit=10)
# dataset.load_data()
# print(dataset.data_df)


# dataset = Newsela(limit=10)
# dataset.load_data()
# print(dataset.data_df)



