class Rule:
    def __init__(self, name:str = "", description:str = "", keys:list[str] = ["quality_metrics"]):
        self.keys = keys
        self.name = name
        self.description = description

    def filter(self, data):
        raise NotImplementedError


class Rule1(Rule):
    def __init__(self):
        super().__init__("Rule 1", "The document must have between 50 and 100,000 words.")

    def filter(self, data):
        metric_data = data["quality_metrics"]
        word_count = metric_data["doc_word_count"]
        return word_count >= 50 and word_count <= 100_000

class Rule2(Rule):
    def __init__(self):
        super().__init__("Rule 2", "The mean word length must be between 3 and 10.")

    def filter(self, data):
        metric_data = data["quality_metrics"]
        mean_word_length = metric_data["doc_mean_word_length"]
        return mean_word_length >= 3 and mean_word_length <= 10

class Rule3(Rule):
    def __init__(self):
        super().__init__("Rule 3", "The symbol to word ratio must be below 0.1.")

    def filter(self, data):
        metric_data = data["quality_metrics"]
        symbol_word_ratio = metric_data["doc_symbol_to_word_ratio"]
        return symbol_word_ratio <= 0.1
    

class Rule4(Rule):
    def __init__(self):
        super().__init__("Rule 4", "90% of lines need to start without a bullet point.")

    def filter(self, data):
        metric_data = data["quality_metrics"]
        n_lines = len(data["lines"])
        n_lines_bulletpoint_start = sum(map(lambda ln: ln[1], metric_data["lines_start_with_bulletpoint"]))
        return n_lines_bulletpoint_start / n_lines <= 0.9
    
class Rule5(Rule):
    def __init__(self):
        super().__init__("Rule 5", "The ratio between characters in the most frequent 2-gram and the total number of characters must be below 0.2.")

    def filter(self, data):
        metric_data = data["quality_metrics"]
        top_2_gram_frac = metric_data["doc_frac_chars_top_2gram"]
        return top_2_gram_frac <= 0.2