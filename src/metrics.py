import os
import pandas as pd
import textstat
from nltk.tokenize import sent_tokenize


class Metrics:

    def __init__(self, input_text):
        self.text = input_text

    def count_words(self) -> int:
        """
        Counts the number of words in the input sentence/text.
        """
        return len(self.text.split())
    
    def count_chars(self) -> int:
        """
        Counts the number of all characters in the input sentence/text.
        Includes both alphanumeric chars and punction, etc.
        """
        return len(self.text)
    
    def count_alphanum(self) -> int:
        """
        Counts the number of alphanumeric characters in the input.
        """   
        count = 0
        for char in self.text:
            if char.isalpha():
                count += 1
        return count

    def count_sents_nltk(self) -> int:
        """
        Counts the number of sentences with NLTK.
        """
        return len(sent_tokenize(self.text))

    def compute_fre(self) -> float:
        """
        Computes Flesh Reading Ease score.
        """
        return textstat.flesch_reading_ease(self.text)

    def compute_ari(self) -> float:
        """
        Computes automated readability index score.
        """
        return textstat.automated_readability_index(self.text) 

    def compute_fkgl(self) -> float:
        """
        Computes the Flesch-Kincaid Grade Level (FKGL) score using textstat.
        """
        return textstat.flesch_kincaid_grade(self.text)

    def compute_dale_chall(self) -> float:
        """
        Computes the Dale-Chall Readability Score. 
        Estimates reading difficulty based on lists of words.
        """
        return textstat.dale_chall_readability_score(self.text)

    def compute_metrics(self) -> dict:
        return {
            'word_count': self.count_words(),
            'char_count': self.count_chars(),
            'alphanum_count': self.count_alphanum(),
            'sent_count_nltk': self.count_sents_nltk(),
            'FRE': self.compute_fre(),
            'ARI': self.compute_ari(),
            'FKGL': self.compute_fkgl(),
            'Dale-Chall': self.compute_dale_chall()
        }
    



