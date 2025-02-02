import os
import pandas as pd
import spacy
# import textstat
# from lexicalrichness import LexicalRichness


class Metrics:

    def __init__(self, input_text, language='en'):
        self.text = input_text
        self.language = language
        # self.data_df = pd.DataFrame()
        if self.language == "en": # default value unless overwritten
            self.nlp = spacy.load("en_core_web_sm")
        elif self.language == "es":
            self.nlp = spacy.load("es_core_web_sm")

    def count_words(self) -> int:
        """
        Count the number of words in the input sentence/text.
        """
        return len(self.text.split())
    
    def count_chars(self) -> int:
        """
        Count the number of all characters in the input sentence/text.
        Includes both alphanumeric chars and punction, etc.
        """
        return len(self.text)
    
    def count_alpha(self) -> int:
        """
        Count the number of alphanumeric characters in the input.
        """   
        count = 0
        for char in self.text:
            if char.isalpha():
                count += 1
        return count
    
    def count_sentences_spacy(self) -> int:
        """
        Count the number of sentences using Spacy.
        """
        nlp = self.nlp
        doc = nlp(self.text)
        return len(list(doc.sents))


text = "Love and peace, but also whatever 1,2,3!! haha@."
metrics = Metrics(text)
print(text)
print(metrics.count_chars())
print(metrics.count_alpha())
print(metrics.count_words())
print(metrics.count_sentences_spacy())

