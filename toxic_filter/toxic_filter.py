from filters.keyword_filter import KeywordFilter
import os
class EnglishToxicsFilter(KeywordFilter):
    def __init__(self, name:str ="", toxic_keywords:list[str] = [] , keys:list[str] = ["text","url"]):
        super().__init__(name =name, keys=keys)

    def filter(self, data:dict ={}):
        return super().filter(data)
    

    def load_internal_data(self, data_file:str):
        toxic_keywords = EnglishToxicsFilter.get_toxic_keywords(data_file)
        self.keywords = toxic_keywords["english"]
        self.regex_pattern = r"\b(" + "|".join(self.keywords) + r")\b"

    @staticmethod
    def get_toxic_keywords(dir:str = "./internal_data/toxic_words"):
        files = os.listdir(dir)
        files = [f for f in files if f.endswith(".txt")]
        toxic_keywords = {}
        for file in files:
            with open(os.path.join(dir, file), 'r') as f:
                language = file.replace(".txt","")
                toxic_words = f.readlines()
                toxic_words = [w.strip() for w in toxic_words]
                toxic_keywords[language] = toxic_words
        return toxic_keywords