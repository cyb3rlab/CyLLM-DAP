from deduplicator.utils import *
from data_managers.data_loader import DataLoader
from data_managers.data_saver import DataSaver
import shutil
class BaseDeduplicator:
    def __init__(self):
        pass

    def deduplicate(self):
        return list(set(self.data))


    def add_data_loader(self, data_loader:DataLoader):
        self.data_loader = data_loader
    
    def add_data_saver(self, data_saver:DataLoader):
        self.data_saver = data_saver

    def loader_register(self, loader):
        self.data_loader = loader
    
    def saver_register(self, saver):
        self.data_saver = saver

class Deduplicator(BaseDeduplicator):
    def __init__(self,  num_workers:int = 1, ngram:int = 5, num_perm:int = 128, threshold:float = 0.5, hash_dir:str = "", saved_dir:str = "", do_cleanup:bool = True, data_loader:DataLoader = None, data_saver:DataSaver = None):
        self.num_workers = num_workers
        self.ngram = ngram
        self.num_perm = num_perm
        self.threshold = threshold
        self.hash_dir = hash_dir
        self.saved_dir = saved_dir
        self.do_cleanup = do_cleanup
        self.data_loader = data_loader
        self.data_saver = data_saver
        super().__init__()

    def deduplicating(self):
        texts_to_minhashes(self.data_loader,self.hash_dir,self.ngram,self.num_perm,self.num_workers)
        hash_path_obj= pathlib.Path(self.hash_dir)
        lsh = lshing_from_disk(hash_path_obj,self.threshold,self.num_perm)
        duplicates = find_duplicates(lsh,hash_path_obj)
        deduplicated_data = deduplicate(self.data_loader.full_load(),duplicates)
        self.data_saver.save(deduplicated_data, file_name="deduplicated_data")
        if self.do_cleanup:
            shutil.rmtree(self.hash_dir)