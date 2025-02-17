
import jsonlines
import datetime
from base.base_filter import BaseFilter
import os
import json
class BaseCollector:
    def __init__(self, name:str = "", des:str = ""):
        self.name = name
        self.des = des
        self.filters = {}

    def run(self):
        raise NotImplementedError
    
    def download(self):
        raise NotImplementedError

    def transform(self, data):
        #transform the original data to the desired format
        #return original data if not implemented
        return data
    def load_done(self):
        if not os.path.exists(self.done_file):
            self.done = []
        else:
            with open(self.done_file,"r") as f:
                self.done = json.load(f)

    def save_done(self):
        with open(self.done_file,"w") as f:
            json.dump(self.done,f)
    def add_filter(self, filter:BaseFilter):
        self.filters[filter.name] = filter

    def apply_filters(self, data):
        #if any filter return True, return True
        for filter in self.filters:
            rs =  filter.run(data)
            if rs: 
                return True
        return False
    @staticmethod
    def id_generator(input:str = ""):
        return str(datetime.datetime.now())+input
    @staticmethod
    def save(data, file_path):
        with jsonlines.open(file_path, 'w') as writer:
            for d in data:
                writer.write(d)
    