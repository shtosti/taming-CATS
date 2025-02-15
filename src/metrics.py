import os
import textstat
import sacrebleu
import evaluate
from evaluate import load
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from comet.models import download_model, load_from_checkpoint

nltk.download('punkt')

class Metrics:
    bertscore_model = None # on first use if ref provided
    comet_model = None 
    sari_model = None

    def __init__(self, input_text, reference_text=None, source_text=None):
        self.text = input_text
        self.reference = reference_text
        self.source = source_text
        self.metrics = {}

    @staticmethod
    def load_bertscore():
        """Loads BERTScore model once for efficiency."""
        if Metrics.bertscore_model is None:
            Metrics.bertscore_model = evaluate.load("bertscore")

    @staticmethod
    def load_sari():
        """Loads SARI model once for efficiency."""
        if Metrics.sari_model is None:
            Metrics.sari_model = evaluate.load("sari")

    @staticmethod
    def load_comet():
        """Loads COMET model once for efficiency."""
        if Metrics.comet_model is None:
            model_path = download_model("Unbabel/wmt22-comet-da")
            Metrics.comet_model = load_from_checkpoint(model_path)

    def count_words(self):
        return len(word_tokenize(self.text))

    def count_chars(self):
        return len(self.text)

    def count_alphanum(self):
        return sum(1 for char in self.text if char.isalpha())

    def count_sents(self):
        return len(sent_tokenize(self.text))

    def compute_fre(self):
        return textstat.flesch_reading_ease(self.text)

    def compute_ari(self):
        return textstat.automated_readability_index(self.text)

    def compute_fkgl(self):
        return textstat.flesch_kincaid_grade(self.text)

    def compute_dale_chall(self):
        return textstat.dale_chall_readability_score(self.text)

    def compute_bleu(self):
        """Computes BLEU score using SacreBLEU."""
        if not self.source:
            return None  # Skip if no source
        return sacrebleu.corpus_bleu([self.text], [[self.source]]).score

    def compute_bertscore(self):
        """Computes BERTScore using evaluate library."""
        if not self.source:
            return None
        self.load_bertscore()
        results = Metrics.bertscore_model.compute(
            predictions=[self.text], references=[self.source], lang="en"
        )
        return results["f1"][0]  # get only first value, F1

    def compute_comet(self):
        """Computes COMET score."""
        if not self.reference:
            return None
        self.load_comet()
        data = [{"src": self.source, "mt": self.text, "ref": self.reference}]
        scores = Metrics.comet_model.predict(data)
        return scores["scores"][0]  # get only first value

    def compute_sari(self):
        """Computes SARI score."""
        if not self.reference or not self.source:
            return None  # SARI requires source (original) and reference (target)
        self.load_sari()
        source = [self.source]
        prediction = [self.text]
        reference = [[self.reference]] # NB expects a list of list
        sari_score = Metrics.sari_model.compute(sources=source, predictions=prediction, references=reference)
        # score = sari.corpus_score([self.text], [[self.reference]], [self.source])
        return sari_score["sari"]  # Extract SARI score

    def compute_metrics(self):
        """Computes all required metrics and returns them as a dictionary."""
        self.metrics = {
            'word_count': self.count_words(),
            'char_count': self.count_chars(),
            'alphanum_count': self.count_alphanum(),
            'sent_count': self.count_sents(),
            'FRE': self.compute_fre(),
            'ARI': self.compute_ari(),
            'FKGL': self.compute_fkgl(),
            'Dale-Chall': self.compute_dale_chall(),
        }

        if self.reference and self.source:
            self.metrics['SARI'] = self.compute_sari()

        if self.source:
            self.metrics.update({
                'BLEU': self.compute_bleu(),
                'BERTScore': self.compute_bertscore(),
                'COMET': self.compute_comet()
            })

        return self.metrics

prediction = "This is a whale."
reference = "That is a killer whale."
source = "This is a bird."
metrics = Metrics(input_text=prediction, reference_text=reference, source_text=source) # no ref, no source
print(metrics.compute_metrics())
print(metrics.text)
print(metrics.count_words())
print(metrics.compute_bertscore())
print(metrics.compute_sari())

