

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from data_managers.data_loader import DataLoader
from data_managers.data_saver import DataSaver
import json
import os
from concurrent.futures import ProcessPoolExecutor,as_completed
def fake_email(x):
        return x.replace("@", "[@]")
    
def fake_url(x):
        return x.replace(".", "[.]")

def fake_phone_number(x):
    newx =  x.replace("1", "2")
    newx = newx.replace("9", "4")
    newx = newx.replace("0", "6")
    newx = newx.replace("5", "7")
    return newx



class Anonymizer:
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    def __init__(self, entities: list = ["PHONE_NUMBER","CREDIT_CARD","EMAIL_ADDRESS", "URL"], language="en", applied_keys: list = ["text"],num_workers:int = 2, threshold:float = 0.5, data_loader:DataLoader = None, data_saver:DataSaver = None):
        self.entities = entities
        self.applied_keys = applied_keys
        self.language = language
        self.num_workers = num_workers
        self.threshold = threshold
        self.data_loader = data_loader
        self.data_saver = data_saver
        self.initize_operator()

    def analyze(self, text):
        return Anonymizer.analyzer.analyze(text=text, entities=self.entities, language=self.language)
    

    def _anonymize(self, text):
        results = self.analyze(text)
        return Anonymizer.anonymizer.anonymize(text=text, analyzer_results=results, operators = self.operators).text
    
    def anonymize(self, data:dict):
        for key in self.applied_keys:
            if key in data:
                data[key] = self._anonymize(data[key])
        return data

    def anonymize_batch(self, samples:list[dict], saved_name:str = ""):
        new_samples = []
        for sample in samples:
            new_samples.append(self.anonymize(sample))
        if saved_name:
            self.data_saver.save(new_samples, saved_name) #data saver will know how to save the data (directory, filetype, etc)
        return new_samples

    def anonymize_multiprocess(self):
        futures = []
        if self.num_workers == 1:
            for id_, data in self.data_loader.iter_load():
                if isinstance(data, list):
                    self.anonymize_batch(data, id_)
                else:
                    data = [data]
                    self.anonymize_batch(data, id_)
        else:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                for id_, data in self.data_loader.iter_load():
                    if isinstance(data, list):
                        futures.append(executor.submit(self.anonymize_batch, data, id_))
                    else:
                        data = [data]
                        futures.append(executor.submit(self.anonymize_batch, data, id_))

    def run(self):
        self.anonymize_multiprocess()
    def initize_operator(self):
        operators = {
            "PHONE_NUMBER": OperatorConfig("custom", {"lambda": fake_phone_number}),
            "CREDIT_CARD": OperatorConfig("replace", {"new_value": ""}),
            "EMAIL_ADDRESS": OperatorConfig("custom", {"lambda": fake_email}),
            "URL": OperatorConfig("custom", {"lambda": fake_url}),
        }
        self.operators = operators


