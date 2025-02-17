

from quality_filters.filter import *
from quality_filters.metrics import *
from quality_filters.rules import *
from filters.keyword_filter import KeywordFilter
from filters.pattern_filter import URLPatternFilter
from collectors.cc_default_collector import CC_WARC_Collector
from collectors.cc_WET_collector import CC_WET_Collector
from collectors.hf_collector import HF_Collector
from collectors.wiki_collector import Wikipedia_Collector
from collectors.s2ocr_collector import S2OCR_Collector
from collectors.book_collector import Book_Collector
from anonymizer.anonymizer import Anonymizer
from data_managers.data_loader import DataLoader
from data_managers.data_saver import DataSaver
class Manager:
    def __init__(self, config:dict = {}):
        self.config = config
        self.current_task = None

        self.tasks = {}


    def add_task(self, task):
        self.tasks[task.task_name] = task



    
    @classmethod
    def init_filter(cls, config:dict = {}):
        class_name = config.get("class_name", "")
        params = config.get("params", {})
        data_file = config.get("data_file", "")
        filter = globals()[class_name](**params)
        filter.load_internal_data(data_file)
        return filter
    

    @classmethod
    def init_collector(cls, config:dict = {}, filters:dict = {}):
        collecter_class_name = config.get("collector", "")
        params = config.get("params", {}) #require base_dir, test_mode, num_workers
        filter_names = config.get("filters", [])
        filters = [filters.get(f, {}) for f in filter_names]
        collector = globals()[collecter_class_name](**params)
        collector.filters = filters
        return collector
    

    @classmethod
    def init_anonymizer(cls, config:dict = {}):
        class_name = config.get("class_name", "")
        params = config.get("params", {})
        anonymizer = globals()[class_name](**params)
        return anonymizer
    
    @classmethod
    def init_loader(cls, config:dict = {}):
        class_name = config.get("class_name", "")
        params = config.get("params", {})
        loader = globals()[class_name](**params)
        return loader
    

    @classmethod
    def init_saver(cls, config:dict = {}):
        class_name = config.get("class_name", "")
        params = config.get("params", {})
        saver = globals()[class_name](**params)
        return saver
    

    @classmethod
    def init_filter(cls, config:dict = {}):
        class_name = config.get("class_name", "")
        params = config.get("params", {})
        filter = globals()[class_name](**params)
        return filter
    


    @classmethod
    def init_metric(cls, class_name:str, params:dict = {}):
        metric = globals()[class_name](**params)
        return metric
    

    @classmethod
    def init_rule(cls, class_name:str, params:dict = {}):
        rule = globals()[class_name](**params)
        return rule