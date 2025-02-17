
from ftlangdetect import detect

def check_language(text, language = ['en']):
    """
    Detects the language of the text and returns True if it is in the language list, False otherwise.
    """
    if not isinstance(text, str):
        return False
    
    try:
        result = detect(text=text, low_memory=False)
    except:
        return False
    
    if result["lang"] in language:
        return True
    
    return False
from base.base_filter import BaseFilter
class LanguageFilter(BaseFilter):
    def __init__(self,name:str ="",  keys:list[str] = [], language:list[str] = ['en']):
        super().__init__(name =name, keys=keys)
        self.language = language


    def run(self, data:dict ={}):
        for key in self.keys:
            if key in data:
                text = data.get(key, "")
                return check_language(text, self.language)
