
from base.base_collector import BaseCollector
import os
import json
from time import sleep
import time
import gzip
import shutil
import concurrent.futures
from itertools import islice
from datasets import load_dataset
class HF_Collector(BaseCollector):
    def __init__(self,name: str = "", base_dir:str ="", num_workers:int =1, test_mode:bool = False, group_size:int =1000, dataset_config:dict = {}, api_key:str = ""):
        super().__init__(name, des="Huggingface Dataset Collector")
        self.base_dir = base_dir
        self.num_workers = num_workers
        self.api_key = api_key
        self.download_dir = os.path.join(self.base_dir,"download")
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
        self.temp_dir = os.path.join(self.base_dir,"temp")
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        self.done_file = os.path.join(self.base_dir,"done_file.json")
        self.output_dir = os.path.join(self.base_dir,"output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.test_mode = test_mode
        self.group_size = group_size
        self.dataset_config = dataset_config
    
    def load_done(self):
        if not os.path.exists(self.done_file):
            self.done = {}
        else:
            with open(self.done_file,"r") as f:
                self.done = json.load(f)

    def run(self):
        self.load_done()
        if self.num_workers == 1:
            count = 0
            for data in self.download():
                count += 1
                file_path = os.path.join(self.output_dir, f"{str(count)}.jsonl")
                self.analyze(data, file_path)
                
                if self.test_mode:
                    break
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:   
                count = 0
                for data in self.download():
                    count += 1
                    file_path = os.path.join(self.output_dir, f"{str(count)}.jsonl")
                    executor.submit(self.analyze, data, file_path)
                    if self.test_mode:
                        break
    
    def get_dataset(self):
        """
        Load the dataset from the huggingface hub
        Args: dataset_config: dict: configuration for the dataset
        Returns: dataset iterator
        """
        dataset_config = self.dataset_config
        ds_iterator = load_dataset(**dataset_config)
        return ds_iterator


    def transform(self, sample):
        new_sample = {}
        new_sample["text"] = sample["raw_content"]
        new_sample["url"] = json.loads(sample["meta"])["url"]
        new_sample["quality_signals"] = sample["quality_signals"]
        new_sample["id_"] = BaseCollector.id_generator("_HF")
        return new_sample

    def download(self):
        if self.api_key:
            try:
                import huggingface_hub
                huggingface_hub.login(self.api_key)
            except:
                pass
        self.iterator = self.get_dataset()
        # dataset = islice(self.iterator, self.group_size)
        # for d in dataset:
        #     yield d
        data = []
        count = 0
        for sample in self.iterator:
            count += 1
            new_sample = self.transform(sample)
            if "url" in new_sample and  new_sample["url"] in self.done:
                continue
            else:
                self.done[new_sample["url"]] = 1
                if count % 1000 == 0:
                    self.save_done()
            data.append(new_sample)
            if len(data) == self.group_size:
                yield data
                data = []
        yield data

    def analyze(self, input_data, saved_name:str = ""):
        filterata = []
        for sample in input_data:
            # new_sample = {}
            # new_sample["text"] = sample["raw_content"]
            # new_sample["url"] = json.loads(sample["meta"])["url"]
            # new_sample["quality_signals"] = sample["quality_signals"]
            # new_sample["id_"] = BaseCollector.id_generator("_HF")
            # if self.apply_filters(new_sample):
            #     filterata.append(new_sample)
            if self.apply_filters(sample):
                filterata.append(sample)
        if len(filterata) > 0:
            BaseCollector.save(filterata, saved_name)








    

