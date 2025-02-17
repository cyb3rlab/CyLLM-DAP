from manager import Manager
from filters.keyword_filter import KeywordFilter
import os
filter_config = { 
                 "class_name":"KeywordFilter",
                 "data_file": "internal_data/keywords.txt",
                 "params": {
                    "keys": ["url"], "name":"filter1", 
                 }
                 }

filter_config2 = { 
                 "class_name":"KeywordFilter",
                 "data_file": "internal_data/keywords.txt",
                 "params": {
                    "keys": ["title"], "name":"filter2", 
                 }
                 }
filter_config3 = { 
                 "class_name":"URLPatternFilter",
                 "data_file": "internal_data/url_patterns.json",
                 "params": {
                    "keys": ["url"], 
                    "name":"filter3", 
                 }
                 }
filter = Manager.init_filter(filter_config)
filter2 = Manager.init_filter(filter_config2)
filter3 = Manager.init_filter(filter_config3)
filters = {}
filters[filter.name] = filter
filters[filter2.name] = filter2
filters[filter3.name] = filter3

print(filter)

commoncrawl_config = {
    
    "collector":"CC_WET_Collector",
    "params": {
        "base_dir":"workspace/commoncrawl",
        "test_mode":True,
        "num_workers":1,
        "name":"commoncrawl",
    },
    "filters": ["filter1", "filter3"]
}


snapshot_file ="internal_data/redpj_snapshots.txt"
with open(snapshot_file) as f:
    snapshots = f.readlines()
    snapshots = [s.strip() for s in snapshots]
hf_config = {
    
    "collector":"HF_Collector",
    "params": {
        "base_dir":"workspace/hf",
        "test_mode":False,
        "num_workers":2,
        "name":"redfine",
        "api_key": "",# need update this api_key
        "dataset_config": {
                "path": "togethercomputer/RedPajama-Data-V2",
                "snapshots": snapshots,
                "languages": ["en"],
                "split": "train",
                "name": "default",
                "streaming": True
            },
        "group_size": 10000
    },
    "filters": ["filter1"]
}


def test_cc_collector():
    collector = Manager.init_collector(commoncrawl_config, filters)
    if not os.path.exists(collector.path_file):
        collector.read_paths()
    collector.run()



def test_hf_collector():
    collector = Manager.init_collector(hf_config, filters)
    collector.run()

wiki_config = {
    
    "collector":"Wikipedia_Collector",
    "params": {
        "base_dir":"workspace/wiki",
        "test_mode":True,
        "num_workers":2,
        "api_key": "", # need update this api_key
        "dataset_config": {
            "path" : "wikimedia/wikipedia",
            "name" :"20231101.en",
            "split" : "train",
            "streaming" : True },
        "group_size": 100
    },
    "filters": ["filter1"]
}
def test_wiki_collector():
    collector = Manager.init_collector(wiki_config, filters)
    collector.run()

s2ocr_config = {
    
    "collector":"S2OCR_Collector",
    "params": {
        "test_mode": True,
        "num_workers": 1,
        "base_dir": "workspace/s2ocr",
        "api_key":"",# need update this api_key
        "download_limit": 5,
    },
    "filters": ["filter2"]
}

def test_s2ocr_collector():
    collector = Manager.init_collector(s2ocr_config, filters)
    # collector.start_bulk_search(["Machine Learning"])
    collector.run()




book_config = {
    
    "collector":"Book_Collector",
    "params": {
        "name": "book",
        "test_mode": True,
        "num_workers": 1,
        "base_dir": "workspace/book",
        "keyword_file": "internal_data/keywords.txt",
        "search_config":{
                "language": "English",
                "extension": "pdf",
                "search_limit": 5
            },
    },
    "filters": []
}


def test_book_collector():
    collector = Manager.init_collector(book_config, filters)
    # collector.start_bulk_search(["Machine Learning"])
    collector.run()
