import random
import os
import json
import pandas as pd
import jsonlines
from datetime import date
import datetime
class DataLoader:
    def __init__(self, source_name:str = "", data_dir:str = "", percentage:float = 10,  doned:dict = {}, batch_size:int = 1, file_type:str = "jsonl"):
        self.source_name = source_name
        self.path = data_dir
        self.data = []
        # self.id_required = id_required
        self.doned = doned
        self.batch_size = batch_size

    def init_range(self, start:int = 0, percentage:float = 10):
        self.start = start
        self.end = start + percentage

    def check_range(self, index:int = 0) -> bool:
        if index >= self.start and index < self.end:
            return True
        return False
    

    def get_data(self, index:int = 0):
        if index < len(self.data):
            return self.data[index]
        else:
            return random.choice(self.data)
    
    def get_random_data(self):
        return random.choice(self.data)
    # def iter_load(self):
    #     if self.batch_size > 1:
    #         return self.iter_load_batch()
    #     else:
    #         return self.iter_load_single()
        
    def iter_load(self):
        if self.batch_size > 1:
            data_batch = []
            for item in DataLoader.dir_iterator(self.path):
                data_batch.append(item)
                if len(data_batch) == self.batch_size:
                    
                    batch_id = str(datetime.datetime.now())+ "_" + self.source_name + "_"

                    yield batch_id, data_batch
                    data_batch = []
        else:
            # if self.id_required:
            for item in DataLoader.dir_iterator(self.path):
                if "id_" in item:
                    yield item["id_"], item #yield each item each time
            # else:
            #     for item in DataLoader.dir_iterator(self.path):
            #         yield item

    def full_load(self):
        if self.id_required:
            data = {}
            for item in DataLoader.file_iterator(self.path):
                if "id_" in item:
                    data[item["id_"]] = item
            return data
        else:
            data = []
            for data in DataLoader.dir_iterator(self.path):
                return data


    def load(self, path:str = "", id_required:bool = False, doned:dict = {}, full:bool = False):
        if full:
            return self.full_load(path, id_required, doned)
        else:
            return self.iter_load(path, id_required, doned)


    @classmethod
    def file_iterator(cls, path:str = ""):
        if path.endswith(".json"):
            for data in DataLoader.get_data_json(path):
                yield data
        if path.endswith(".csv"):
            for data in DataLoader.get_data_csv(path):
                yield data
        if path.endswith(".jsonl"):
            for data in DataLoader.get_data_jsonline(path):
                yield data
        
    @classmethod
    def dir_iterator(cls, dir:str = "", file_type:str = "jsonl"):
        for file in os.listdir(dir):
            if file_type == "any":
                if file.endswith(".json"):
                    file_path = os.path.join(dir, file)
                    for data in DataLoader.get_data_json(file_path):
                        yield data
                if file.endswith(".csv"):
                    file_path = os.path.join(dir, file)
                    for data in DataLoader.get_data_csv(file_path):
                        yield data
                if file.endswith(".jsonl"):
                    file_path = os.path.join(dir, file)
                    for data in DataLoader.get_data_jsonline(file_path):
                        yield data
            else:
                if file_type == "json":
                    if file.endswith(".json"):
                        file_path = os.path.join(dir, file)
                        for data in DataLoader.get_data_json(file_path):
                            yield data
                if file_type == "csv":
                    if file.endswith(".csv"):
                        file_path = os.path.join(dir, file)
                        for data in DataLoader.get_data_csv(file_path):
                            yield data
                if file_type == "jsonl":
                    if file.endswith(".jsonl"):
                        file_path = os.path.join(dir, file)
                        for data in DataLoader.get_data_jsonline(file_path):
                            yield data



    @classmethod
    def get_data_jsonline(cls, file_path:str = ""):
        with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    yield data
    
    @classmethod
    def get_data_json(cls, file_path:str = ""):
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    yield item
            if isinstance(data, dict):
                yield data

    @classmethod
    def get_data_csv(cls, file_path:str = ""):
        data = pd.read_csv(file_path)
        for item in data:
            yield item

    def write_data(self, data, file_path):
        pass
    
