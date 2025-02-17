import os
import json
import subprocess
import time
import pandas as csv
import gzip
import shutil
import psutil
import concurrent.futures
import random
from warc3 import warc
from time import sleep


# copy ="aws s3 cp s3://commoncrawl/crawl-data/{cc}/segments/{segment}/wet/{file} {local_path}"
# copy2 = "aws s3 cp {remote_path} {local_path}"

# def download(remote="", local =""):
#     command = f"aws s3 cp {remote} {local}"
#     file_path = local
#     while(not os.path.isfile(file_path)):
#         os.system(command)
#         time.sleep(0.1)

from collectors.cc_default_collector import CC_WARC_Collector
from base.base_collector import BaseCollector
def check_year(paths, years:list = []):
    if len(years) == 0:
        return paths
    new_paths = []
    for p in paths:
        for y in years:
            check_str =  f"CC-MAIN-{str(y)}"
            if check_str in p["remote_path"]:
                new_paths.append(p)
    return new_paths

class CC_WET_Collector(CC_WARC_Collector):
    def __init__(self, name:str ="",  base_dir:str ="", test_mode:bool = True, num_workers:int =2, download_limit:int = 5):
        super().__init__(name, base_dir, test_mode, num_workers, download_limit)
        self.cc_type = "wet"



    def analyze(self, input_path:str = ""):
        print(f"start analyze {input_path}")
        file_name = input_path.split("/")[-1] #get the file name
        if file_name.endswith(".warc.wet.gz"):
            unzip_name = file_name.replace(".warc.wet.gz",".warc")
        else:
            unzip_name = file_name.replace(".warc.gz",".warc")
        output_name = file_name + ".jsonl"
        unzip_path = os.path.join(self.temp_dir,unzip_name)
        #unzip the file
        with gzip.open(input_path, 'rb') as f_in:
            with open(unzip_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        #analyze the file
        data_file = os.path.join(self.output_dir, output_name)
        data = []
        with warc.open(unzip_path) as f:
            for i,record in enumerate(f): 
                url = record.url
                if url is None:
                    continue
                text = record.payload.read().decode("utf-8")

                _data = {"text":text, "url":url, "id_": BaseCollector.id_generator("_WET")}
                rs = self.apply_filters(_data)
                if rs:
                    data.append(_data)
        print(f"done analyze {input_path}")
        BaseCollector.save(data, data_file)
        os.remove(input_path)#delete the original file
        os.remove(unzip_path)#delete the unzipped file
        print(f"done analyze and delete {input_path} \n unzip {unzip_path} \n save {data_file}")


