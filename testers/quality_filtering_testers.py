save_config = {
    "class_name":"DataSaver",
    "params": {
        "data_dir":"workspace/quality",
        "file_type":"jsonl"
    }
}
loader_config = {
    "class_name":"DataLoader",
    "params": {
        "source_name":"commoncrawl",
        "data_dir":"workspace/commoncrawl/output",
        "file_type":"jsonl",
        "batch_size": 1000
    }
}
quality_config = {
    "class_name":"QualityFilter",
    "params": {
        "metrics": ["M1","M2","M3","M4","M5","M6","M7","M8","M9","M10","M11","M12", "M13", "M14", "M15", "M16", "M17"],
        "rules": ["Rule1","Rule2","Rule3","Rule4","Rule5"],
        "metrics_save": True,
        "num_workers": 1
    },


}

from manager import Manager
from data_managers.data_loader import DataLoader
from data_managers.data_saver import DataSaver

def test_quality_filter():
    loader = Manager.init_loader(loader_config)
    saver = Manager.init_saver(save_config)
    metrics = {} 
    for m in quality_config["params"]["metrics"]:
        metrics[m] = Manager.init_metric(m)
    rules = {}
    for r in quality_config["params"]["rules"]:
        rules[r] = Manager.init_rule(r)
    quality_config["params"]["metrics"] = metrics
    quality_config["params"]["rules"] = rules
    quality_config["params"]["data_loader"] = loader
    quality_config["params"]["data_saver"] = saver
    quality_filter = Manager.init_filter(quality_config)
    quality_filter.run_multiprocess()
    print(1)

