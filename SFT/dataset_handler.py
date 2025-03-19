# from datasets import load_dataset

# class DatasetHandler:
#     def __init__(self, file_path, format="csv"):
#         self.file_path = file_path
#         self.format = format
#         self.dataset = None

#     def load_data(self, split_ratio=0.1):
#         self.dataset = load_dataset(self.format, data_files=self.file_path)
#         self.dataset = self.dataset['train'].train_test_split(test_size=split_ratio)

#     def get_datasets(self):
#         return self.dataset['train'], self.dataset['test']