from pathlib import Path
import math
from stop_words import get_stop_words
import operator
from collections import Counter
import re
import numpy as np
PRECISION = 3

class BaseMetric():
    r""" Base class for all metrics. """
    __slots__ = ["name"]
    def __init__(self, name: str = ""):
        self.name = name
    def __call__(self, document: dict = {}) :
        raise NotImplementedError("This method must be implemented in the subclass.")
    
class M1(BaseMetric):

    SEARCH_TEXT = "lorem ipsum"
    SEARCH_REGEX = re.compile(r"lorem ipsum", re.IGNORECASE)

    __slots__ = ()
    def __init__(self):
        super().__init__("lorem_ipsum_ratio")

    def __call__(self, document: dict = {}):
        if len(document["text"]) == 0:
            return [(0, len(document), 0.0)]

        if self.SEARCH_TEXT not in document["text"]:
            return [(0, len(document), .0)]

        num_occurences = len(self.SEARCH_REGEX.findall(
            document["text"]
        ))

        score = float(num_occurences) / len(document["text"])
        score = round(score, PRECISION)

        return score



class M2(BaseMetric):  
    r""" Whether the lines that start with a bullet point symbol. The
    following set of unicodes are considered a bullet point:
    \u2022 (bullet point), \u2023 (triangular bullet point), \u25B6 (black
    right pointing triangle), \u25C0 (black left pointing triangle),
    \u25E6 (white bullet point), \u25A0 (black square), \u25A1 (white
    square), \u25AA (black small square), \u25AB (white small square),
    \u2013 (en dash)."""
    BULLET_POINT_SYMBOLS = (
        "\u2022",  # bullet point
        "\u2023",  # triangular bullet point
        "\u25B6",  # black right pointing triangle
        "\u25C0",  # black left pointing triangle
        "\u25E6",  # white bullet point
        "\u25A0",  # black square
        "\u25A1",  # white square
        "\u25AA",  # black small square
        "\u25AB",  # white small square
        "\u2013",  # en dash
    )

    __slots__ = ()
    def __init__(self):
        super().__init__("lines_start_with_bulletpoint_ratio")

    def __call__(self, document:dict = {}) -> float:
        if "lines" not in document:
            return 0.0
        count = 0
        for line in document["lines"]:
            if line.lstrip().startswith(self.BULLET_POINT_SYMBOLS):
                count += 1

        return round( count / len(document["lines"]), 3)


class M3(BaseMetric):  
    r""" The ratio between number of numerical characters and total number of
    characters in each line. This is based on text after lowercasing and
    removing punctuation."""
    __slots__ = ()
    def __init__(self):
        super().__init__("lines_numerical_chars_fraction")

    def _process(self, line:str):  
        if len(line) == 0:
            return 0.0

        score = sum(map(str.isnumeric, line)) / len(line)
        score = round(score, PRECISION)
        return score

    def __call__(self, document:dict = {} ) :
        results =  []
        for i in range(len(document["lines"])):
            results.append((i, self._process(document["lines"][i])))
        return results


class M4(BaseMetric):  
    r""" The number of occurences of the word "javascript" in each line. """
    SEARCH_TEXT = "javascript"
    __slots__ = ()

    def __init__(self):
        super().__init__("lines_javascript_counts")

    def _process(self, line):
        if len(line) == 0:
            return 0.0

        score = float(sum(
            1 for w in line.split() if w == self.SEARCH_TEXT
        ))

        return score

    def __call__(self, document: dict = {} ) :
        results =  []
        for i in range(len(document["lines"])):
            results.append((i, self._process(document["lines"][i])))
        return results


class M5(BaseMetric):  
    r""" A list of integers indicating whether (1) or not (0) a line ends with
    a terminal punctuation mark. A terminal punctation mark is defined as
    one of the following: ".", "!", "?", "”" """
    TERMINAL_PUNCTUATION_MARKS = (".", "!", "?", "”")
    __slots__ = ()

    def __init__(self):
        super().__init__("lines_ending_with_terminal_punctuation_mark")

    def _process(self, line:str = ""):
        score = line.rstrip().endswith(
            self.TERMINAL_PUNCTUATION_MARKS
        )
        score = float(score)
        return score

    def __call__(self, document: dict = {}) :
        results =  []
        for i in range(len(document["lines"])):
            results.append((i, self._process(document["lines"][i])))
        return results


class M6(BaseMetric):  
    r""" The number of words in each line. This is computed based on the
    normalied text. Normalization is done by lowercasing the text and
    removing punctuation."""
    __slots__ = ()

    def __init__(self):
        super().__init__("lines_num_words")

    def _process(self, line:str = ""):  
        score = len(line.split())
        return score

    def __call__(self, document: dict = {}) :
        results =  []
        for i in range(len(document["lines"])):
            results.append((i, self._process(document["lines"][i])))
        return results



class M7(BaseMetric):
    r""" The ratio between number of uppercase letters and total number of
    characters in each line. This is based on the raw text. """
    __slots__ = ()

    def __init__(self):
        super().__init__("lines_uppercase_letter_fraction")

    def _process(self, line:str = ""):
        if len(line) == 0:
            return 0

        score = sum(map(str.isupper, line)) / len(line)
        score = round(score, PRECISION)
        return score

    def __call__(self, document: dict = {}) :
        results =  []
        for i in range(len(document["lines"])):
            results.append((i, self._process(document["lines"][i])))
        return results
    

class M8(BaseMetric):
    r""" The ratio between the number of stop words and the number of words in
    the document. """
    __slots__ = ["_stop_words"]

    def __init__(self):
        super().__init__("doc_stop_word_fraction")
        self._stop_words = get_stop_words("en")

    def __call__(self, document: dict = {}) :
        if len(document["words"]) == 0:
            return 0.0

        num_stop_words = sum(
            map(lambda w: w in self._stop_words, document["words"])
        )

        score = float(num_stop_words) / len(document["words"])
        score = round(score, PRECISION)

        return score


class M9(BaseMetric):  
    r""" The ratio between the number of occurences of '{' or '}' and the
    number of characters in the raw text. """
    SEARCH_TEXT = ("{", "}")
    __slots__ = ()

    def __init__(self):
        super().__init__("doc_curly_bracket_ratio")

    def __call__(self, document: dict = {}) :
        if len(document["text"]) == 0:
            return 0.0

        if all(map(lambda x: x not in document["text"], self.SEARCH_TEXT)):
            return  0.0

        num_occurences = sum(
            map(lambda x: operator.countOf(document["text"], x),
                self.SEARCH_TEXT)
        )

        score = float(num_occurences) / len(document["text"])
        score = round(score, PRECISION)

        return score






class M10(BaseMetric): 
    r""" The number of sentences in the content. This is calculated using
    the regex r'\b[^.!?]+[.!?]*' """
    SENT_PATTERN = re.compile(r'\b[^.!?]+[.!?]*', flags=re.UNICODE)

    __slots__ = ()

    def __init__(self):
        super().__init__("doc_num_sentences")
    def __call__(self, document: dict = {}) :
        r""" count the number of sentences in the content using regex"""
        score = float(len(self.SENT_PATTERN.findall(document["text"])))
        return score


class M11(BaseMetric):
    r""" The number of words in the content after normalization. """
    __slots__ = ()

    def __init__(self):
        super().__init__("doc_word_count")

    def __call__(self, document: dict = {}) :
        return len(document["words"])


class M12(BaseMetric): 
    r""" The mean length of words in the content normalization. """
    __slots__ = ()

    def __init__(self):
        super().__init__("doc_mean_word_length")

    def __call__(self, document: dict = {}) :
        num_tokens = len(document["words"])
        if len(document["words"]) == 0:
            return 0.0

        num_chars = float(sum(map(len, document["words"])))
        score = num_chars / num_tokens
        score = round(score, PRECISION)
        return score


class M13(BaseMetric):  
    r""" The ratio of symbols to words in the content. This is analogous to
    the signal used in Gopher. Symbols are defined "#", "...", and "…". """
    SYMBOLS = ("#", "...", "…")

    __slots__ = ()

    def __init__(self):
        super().__init__("doc_symbol_to_word_ratio")

    def __call__(self, document: dict = {}) :
        num_words = len(document["words"])

        if num_words == 0:
            return [(0, len(document), None)]

        # count the number of symbols in the content
        num_symbols = float(sum(
            document["text"].count(x) for x in self.SYMBOLS
        ))

        score = num_symbols / num_words
        score = round(score, PRECISION)
        return score


class M14(BaseMetric):  
    r""" The fraction of lines that end with an ellipsis, where an ellipsis
    is defined as either "..." or "…". """
    ELLIPSIS_SYMBOLS = ("...", "…")

    __slots__ = ()

    def __init__(self):
        super().__init__("doc_frac_lines_end_with_ellipsis")

    def __call__(self, document: dict = {}) :
        num_lines = len(document["lines"])

        if num_lines == 0:
            return 0.0

        total_ellipsis_lines = float(sum(
            line.rstrip().endswith(self.ELLIPSIS_SYMBOLS)
            for line in document["lines"]
        ))

        score = total_ellipsis_lines / num_lines
        score = round(score, PRECISION)
        return score


class M15(BaseMetric):  
    r""" The fraction of words that contain no alphabetical character.
    This is based on the raw content. """
    ALPH_REGEX = re.compile(r"[a-zA-Z]")

    __slots__ = ()

    def __init__(self):
        super().__init__("doc_frac_no_alph_words")

    def __call__(self, document: dict = {}) :
        num_words = len(document["words"])

        if num_words == 0:
            return 0.0

        num_words_with_alpha = float(sum(
            int(self.ALPH_REGEX.search(word) is not None)
            for word in document["words"]
        ))

        score = 1.0 - num_words_with_alpha / num_words
        score = round(score, PRECISION)
        return score


class M16(BaseMetric):  
    r""" The fraction of unique words in the content. This is also known as
    the degeneracy of a text sample. Calculated based on the normalized
    content. """
    __slots__ = ()

    def __init__(self):
        super().__init__("doc_frac_nnique_words")

    def __call__(self, document: dict = {}) :
        num_words = len(document["words"])

        if num_words == 0:
            0.0

        score = float(len(set(document["words"]))) / num_words
        score = round(score, PRECISION)
        return score


class M17(BaseMetric):  
    r""" The entropy of the unigram distribution of the
    content. This measures the diversity of the content and is computed
    using sum(-x / total * log(x / total)) where the sum is taken over
    over counts of unique words in the noramlized (punctuation removed,
    lowercased) content."""
    __slots__ = ()

    def __init__(self):
        super().__init__("doc_unigram_entropy")

    def __call__(self, document: dict = {}) :
        if len(document["words"]) == 0:
            0.0

        # count the number of times each word appears in the content
        counter = Counter(document["words"])

        # calculate the entropy of the unigram distribution
        total = sum(counter.values())
        entropy = sum(map(
            lambda x: -x / total * math.log(x / total) if x > 0 else 0.0,
            counter.values()
        ))

        score = round(entropy, PRECISION)
        return score


class M18(BaseMetric):  
    r""" The fraction of words in the content that only conist of uppercase
    letters. This is based on the raw content."""
    __slots__ = ()

    def __init__(self):
        super().__init__("doc_frac_all_caps_words")
    def __call__(self, document: dict = {}) :
        num_words = len(document["words"])

        if num_words == 0:
            return 0.0

        score = float(sum(map(str.isupper, document["words"]))) / num_words
        score = round(score, PRECISION)
        return score