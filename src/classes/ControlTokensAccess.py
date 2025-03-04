
class ControlTokensAccess:
    """Class to generate control tokens in the ACCESS format"""
    
    def __init__(self, source_text: str = None, target_text: str = None):
        self.source_text = source_text or ""  # default to empty string if None
        self.target_text = target_text or "" # ~

        self.num_chars = self.compute_char_compression()
        self.lev_sim = self.compute_levenshtein_similarity()
        self.word_freq = self.compute_word_frequency()
        self.dep_tree_depth = self.compute_dependency_tree_depth()

    def compute_char_compression(self):
        """Computes character-level compression rate."""
        if len(self.source_text) > 0:
            pass # TODO
        return 0.0

    def compute_levenshtein_similarity(self):
        """Computes Levenshtein similarity."""
        if self.source_text and self.target_text:
            pass # TODO
        return 0.0

    def compute_word_frequency(self):
        """Computes word frequency"""
        if self.source_text and self.target_text:
            pass # TODO
        return 0.0

    def compute_dependency_tree_depth(self):
        """Computes dependency tree depth."""
        if self.source_text and self.target_text:
            pass # TODO
        return 0.0

    def __str__(self):
        """Returns formatted control tokens as strings."""
        return f"<NbChars_{self.num_chars}> <LevSim_{self.lev_sim}> <WordFreq_{self.word_freq}> <DepTreeDepth_{self.dep_tree_depth}>"

    def as_dict(self):
        """Returns all control tokens as a dict."""
        return {
            "NbChars": f"<NbChars_{self.num_chars}>",
            "LevSim": f"<LevSim_{self.lev_sim}>",
            "WordFreq": f"<WordFreq_{self.word_freq}>",
            "DepTreeDepth": f"<DepTreeDepth_{self.dep_tree_depth}>"
        }


# tokens = ControlTokensAccess("This is a long sentence.", "This is short.")
# print("with input:")
# print(tokens)  # String output
# print(tokens.as_dict())  # Dictionary output

# tokens_none = ControlTokensAccess()  # No input texts
# print("\nno input:")
# print(tokens_none)
# print(tokens_none.as_dict())
