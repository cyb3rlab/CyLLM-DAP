from quality_filters.metrics import *
from quality_filters.rules import Rule
from base.base_filter import BaseFilter
from concurrent.futures import ProcessPoolExecutor,as_completed
import os
from data_managers.data_loader import DataLoader
from data_managers.data_saver import DataSaver
# nlp = nlp.load("en_core_web_sm")
sentence_regex = re.compile(r'(?<=[.!?:])\s+')
word_regex = re.compile(r"\b\w+(?:['’]\w+)?\b")

def analyzing(text):
    # split the text into sents and words
    # doc = nlp(text)
    # sents = []
    # words = []
    # for token in doc:
    #     words.append(token.text)
    # for sent in doc.sents:
    #     sents.append(sent.text)
    sentences = sentence_regex.split(text)
    words = []
    for sent in sentences:
        _words = word_regex.findall(sent)
        words.extend(_words)
    
    return {"lines": sentences, "words": words}


class QualityFilter(BaseFilter):
    def __init__(self, metrics:dict = {}, rules:dict = {}, keys: list = ["text"], num_workers:int = 1,  metrics_save:bool = False, data_loader:DataLoader = None, data_saver:DataSaver = None):
        self.metrics = metrics
        self.rules = rules
        self.keys = keys
        self.num_workers = num_workers
        self.data_loader = data_loader
        self.data_saver = data_saver
        self.metrics_save = metrics_save

    def add_metrics(self, metric: BaseMetric):
        self.metrics[metric.__class__.__name__] = metric

    def calculate_metrics(self, data:dict = {}):
        data = self.add_required_data(data) # add lines and words to data

        metric_data = {}
        for metric_name, metric in self.metrics.items():
            rs =  metric(data)
            metric_data[metric.name] = rs
        data["quality_metrics"] = metric_data
        return data
    
    def add_required_data(self, data):
        data.update(analyzing(data["text"]))
        return data


    def add_rules(self, rule: Rule):
        self.rules[rule.name] = rule
    


    def run_batch(self, samples:list[dict], saved_name:str = ""):
        new_samples = []
        for sample in samples:
            rs = self.run(sample)
            if rs:
                new_samples.append(rs)
        if saved_name:
            if len(new_samples) == 0:
                return
            self.data_saver.save(new_samples, saved_name) #data saver will know how to save the data (directory, filetype, etc)
        return new_samples
            
    def run(self, data): 
        # calculate the metrics
        if "quality_metrics" not in data:
            data = self.calculate_metrics(data)
        # apply the rules
        for rule_name, rule in self.rules.items():
            rs = rule.filter(data)
            if rs:

                if not self.metrics_save: # not save the metrics data
                    del data["quality_metrics"]
                    del data["lines"]
                    del data["words"]
                    # self.data_saver.save(data, saved_name) #data saver will know how to save the data (directory, filetype, etc)
                return data
        return False
    
    def run_multiprocess(self):
        futures = []
        if self.num_workers == 1:
            for id_, data in self.data_loader.iter_load():
                if isinstance(data, list):
                    self.run_batch(data, id_)
                else:
                    data = [data]
                    self.run_batch(data, id_)
        else:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                for id_, data in self.data_loader.iter_load():
                    if isinstance(data, list):
                        futures.append(executor.submit(self.run_batch, data, id_))
                    else:
                        data = [data]
                        futures.append(executor.submit(self.run_batch, data, id_))