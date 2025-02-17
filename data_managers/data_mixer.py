import os
import json
import pandas as pd
import jsonlines
from datasets import load_dataset
import datasets
import random
from data_managers.data_loader import DataLoader
class DataMixer:
    def __init__(self, data_sources:dict = {}, saved_path:str = "", max_size:int = 1000):
        self.data_sources = data_sources
        self.saved_path = saved_path
        self.max_size = max_size

    def fetch_data(self):
        data_sources = {}
        for source in self.data_sources:
            source_name = source.get("source", "")
            source_dir = source.get("dir", "")
            percentage = source.get("percentage", 0)
            datasource = DataLoader(source_name, source_dir, percentage)
            datasource.load_data(source_dir)
            data_sources[source_name] = datasource
        self.data_sources = data_sources


    def mix_data(self):
        total_percentage = 0
        for source in self.data_sources:
            total_percentage += source.get("percentage", 0)
        if total_percentage != 100:
            print("the total percentage should be 100")
            return
        start = 0
        record = {}
        for k,v in self.data_sources.items():
            v.init_range(start = start, percentage = v.get("percentage", 0))
            record[k] = 0
        flag = False
        data = []
        while not flag:
            random_number = random.randint(0, 99)
            for k,v in self.data_sources.items():
                if v.check_range(random_number):
                    item = v.get_data(record[k])
                    record[k] += 1
                    data.append(item)
                    break
            if len(data) > self.max_size:
                flag = True
    
        with jsonlines.open(self.saved_path, 'w') as writer:
            for d in data:
                writer.write(d)
            
    


    @classmethod
    def uploade_data(cls, path:str = "", repo_id:str = ""):
        dataset = load_dataset("json", data_files=path)
        try:
            dataset.push_to_hub(repo_id, max_shard_size= "2GB")
        except Exception as e:
            print(e)
            print("upload failed")

    

