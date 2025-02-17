
from base.base_filter import BaseFilter
import re
import json
def all_keywords_present(text, keywords):
    return all(
        re.search(re.escape(kw), text, re.IGNORECASE)
        for kw in keywords
    )
class URLPatternFilter(BaseFilter):
    """
    """
    def __init__(self, name:str ="", url_patterns:list =[], keys:list[str] = ["url"]):
        super().__init__(name =name, keys=keys)
        self.url_patterns = url_patterns

    def run(self, data:dict ={})-> bool:
        for key in self.keys:
            if key in data:
                text = data.get(key, "")
                for pattern in self.url_patterns:
                    keywords = pattern.get("keywords", [])
                    check = all_keywords_present(text, keywords)
                    if check:
                        return True
        return False


    def load_internal_data(self, data_file:str):
        with open(data_file, 'r') as f:
            data = json.load(f)
            self.url_patterns = data

class RegexPatternFilter(BaseFilter):
    def __init__(self, name:str ="",  patterns:list =[], keys:list[str] = ["text"]):
        super().__init__(name =name, keys=keys)
        self.patterns = patterns
        self.keys = keys

    def run(self, data:dict ={})-> bool:
        for key in self.keys:
            if key in data:
                text = data.get(key, "")
                for pattern in self.patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        return True
        return False