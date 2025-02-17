from base.base_filter import BaseFilter
import re
import json
import os
class KeywordFilter(BaseFilter):
    def __init__(self,name:str= "", keywords:list[str] = [] , keys:list[str] = ["url"]):
        super().__init__(name =name, keys=keys)
        self.keywords = keywords
        escaped_keywords = [re.escape(k) + r"[a-z0-9]*" for k in keywords]
        self.regex_pattern = r"(?i)(?:^|[^a-z0-9])(" + "|".join(escaped_keywords) + ")"
        self.regex = re.compile(self.regex_pattern, re.IGNORECASE)

    def load_internal_data(self, data_file:str):
        with open(data_file, 'r') as f:
            keywords = f.readlines()
            keywords = [w.strip() for w in keywords]
            self.keywords = keywords
            escaped_keywords = [re.escape(k) + r"[a-z0-9]*" for k in keywords]
            self.regex_pattern = r"(?i)(?:^|[^a-z0-9])(" + "|".join(escaped_keywords) + ")"
            self.regex = re.compile(self.regex_pattern, re.IGNORECASE)

    def run(self, data:dict ={}):
        for key in self.keys:
            if key in data:
                text = data.get(key, "")
                if key == "url":
                    text = re.sub(r"-|_", " ", text)
                match = self.regex.search(text)
                if match:
                    # print(match.group(0))
                    return True
        return False
    
    @staticmethod
    def get_keywords(dir:str = "./internal_data/keywords"):
        files = os.listdir(dir)
        files = [f for f in files if f.endswith(".txt")]
        keywords = []
        for file in files:
            with open(os.path.join(dir, file), 'r') as f:
                words = f.readlines()
                words = [w.strip() for w in words]
                keywords.extend(words)
        return keywords
        
    
    @staticmethod
    def get_keywords_from_file(file:str = ""):
        with open(file, 'r') as f:
            keywords = f.readlines()
            keywords = [w.strip() for w in keywords]
        return keywords