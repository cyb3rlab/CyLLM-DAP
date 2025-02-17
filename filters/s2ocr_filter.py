from base.base_filter import BaseFilter
import re
import json
import os


class S2ocrIDSearchFilter(BaseFilter):
    def __init__(self, name:str ="", references:dict[str,list] = {}, keys:list[str] = ["external_ids"]):
        #references is a dictionary with keys as the external_id type and values as the list of external_ids
        super().__init__(name =name, keys=keys)
        self.bulk_search_metas = references

    def filter(self, data:dict ={}):
        for key in self.keys:
            if key in data:
                external_ids = data.get(key, {})
                for k, v in external_ids.items():
                    if k in self.bulk_search_metas and v is not None and v in self.bulk_search_metas[k]:
                        # return {k: v}
                        return True
        return False