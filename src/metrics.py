import os
import pandas as pd
import textstat
from nltk.tokenize import sent_tokenize
import sacrebleu
import evaluate
from comet.models import download_model, load_from_checkpoint


class Metrics:

    def __init__(self, input_text, reference_text=None):

        # arguments
        self.text = input_text
        self.reference = reference_text

        # load rouge
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")

        # initialize metrics
        self.word_count = 0
        self.char_count = 0
        self.alphanum_count = 0
        self.sentence_count  = 0
        self.FRE = 0.0
        self.ARI = 0.0
        self.FKGL = 0.0
        self.Dale_Chall = 0.0
        self.BLEU = 0.0
        self.BERTScore = 0.0
        self.COMET = 0.0
        self.ROUGE = 0.0

    def count_words(self) -> int:
        """
        Counts the number of words in the input sentence/text.
        """
        self.word_count = len(self.text.split())
        return self.word_count
    
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
    
    def compute_BLEU(self) -> float:
        """
        Computes BLEU score using SacreBLEU.
        """
        if not self.reference:
            raise ValueError("Reference text is necessary to compute BLEU.")
        bleu = sacrebleu.corpus_bleu(
                                        [self.text], # hypothesis
                                        [[self.reference]] # reference
                                        )
        self.BLEU = bleu.score
        return self.BLEU

    def compute_BERTScore(self) -> float:
        """
        Computes BERTScore using the evaluate library.
        """
        if not self.reference:
            raise ValueError("Reference text is necessary to compute BERTScore.")
        results = self.bertscore.compute(predictions=[self.text], references=[self.reference], lang="en")
        self.BERTScore = results["f1"][0] # extract the first value from the array
        return self.BERTScore


    def compute_COMET(self) -> float:
        """
        Computes the COMET score for the given text and reference.
        """
        if not self.reference:
            raise ValueError("Reference text is necessary to compute COMET.")

        # Load COMET model (downloads if not available)
        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)

        data = [{"src": self.text, "mt": self.text, "ref": self.reference}]

        # Compute scores
        scores = model.predict(data)

        self.COMET = scores["scores"]  # Extract scores
        return self.COMET




    def compute_metrics(self) -> dict:
        metrics = {
            'word_count': self.count_words(),
            'char_count': self.count_chars(),
            'alphanum_count': self.count_alphanum(),
            'sent_count_nltk': self.count_sents_nltk(),
            'FRE': self.compute_fre(),
            'ARI': self.compute_ari(),
            'FKGL': self.compute_fkgl(),
            'Dale-Chall': self.compute_dale_chall()
        }
        if self.reference:
            metrics.update({
                'BLEU': self.compute_bleu(),
                'ROUGE': self.compute_rouge()
            })

        return metrics
    


predictions = "Corneal ulcers cause redness, pain, usually a feeling like a foreign object is in the eye (foreign body sensation), aching, sensitivity to bright light, and increased tear production."
references = "Conjunctival redness, eye ache, foreign body sensation, photophobia, and lacrimation may be minimal initially."
metrics = Metrics(input_text=predictions, reference_text=references)

# metrics.compute_BERTScore()
# print(f"current BERTScore: {metrics.BERTScore}")

print("COMET Score:", metrics.compute_COMET())
print(f"current COMET: {metrics.COMET}")