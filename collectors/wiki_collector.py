from collectors.hf_collector import HF_Collector
from base.base_collector import BaseCollector
import json
from datasets import load_dataset
class Wikipedia_Collector(HF_Collector):
    def __init__(self, name: str = "", base_dir: str = "", num_workers: int = 1, test_mode: bool = False,
                 group_size: int = 1000, dataset_config: dict = {}, api_key: str = ""):
        super().__init__(name, base_dir, num_workers, test_mode, group_size, dataset_config, api_key)


    def get_dataset(self):
        """
        Load the dataset from the huggingface hub
        Args: dataset_config: dict: configuration for the dataset
        """
        ds_iterator = load_dataset(**self.dataset_config)
        # ds_iterator = load_dataset(path="wikimedia/wikipedia", name="20231101.en", streaming=True)
        return ds_iterator

    def transform(self, sample):
        new_sample = {}
        new_sample["text"] = sample["text"]
        new_sample["url"] = sample["url"]
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
        data = []
        count = 0
        for sample in self.iterator:
            count += 1
            new_sample = self.transform(sample)
            if "url" in new_sample and new_sample["url"] in self.done:
                continue
            else:
                self.done[new_sample["url"]] = 1
                if count % 10000 == 0:
                    self.save_done()
            data.append(new_sample)
            if len(data) == self.group_size:
                yield data
                data = []
        yield data