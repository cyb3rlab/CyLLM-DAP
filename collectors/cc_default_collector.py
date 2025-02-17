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
import warcio
from warcio.archiveiterator import ArchiveIterator
from trafilatura import html2txt
from time import sleep

def extract_text_from_html(html):
    data = html2txt(html)
    return data
# copy ="aws s3 cp s3://commoncrawl/crawl-data/{cc}/segments/{segment}/wet/{file} {local_path}"
# copy2 = "aws s3 cp {remote_path} {local_path}"

# def download(remote="", local =""):
#     command = f"aws s3 cp {remote} {local}"
#     file_path = local
#     while(not os.path.isfile(file_path)):
#         os.system(command)
#         time.sleep(0.1)

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


class CC_WARC_Collector(BaseCollector):
    def __init__(self, name:str ="", base_dir:str ="", test_mode:bool = True, num_workers:int =2, download_limit:int =5):
        super().__init__(name, des="Common Crawl WARC Collector")
        self.cc_type ="warc"
        self.base_dir = base_dir
        self.test_mode = test_mode
        self.num_workers = num_workers
        self.main_cc_path = os.path.join(self.base_dir,"main_cc.txt")
        self.segment_path = os.path.join(self.base_dir,"segments.txt")
        self.file_path = os.path.join(self.base_dir,"files.txt")
        self.path_file = os.path.join(self.base_dir,"paths.json")
        self.download_dir = os.path.join(self.base_dir,"download")
        self.done_file = os.path.join(self.base_dir,"done_files.json")
        self.raw_dir = os.path.join(self.base_dir,"raw")
        self.temp_dir = os.path.join(self.base_dir,"temp")
        self.backup_dir = os.path.join(self.base_dir,"backup")
        self.download_limit = download_limit
        self.filters = {}
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        self.output_dir = os.path.join(self.base_dir,"output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_paths(self, years:list = []):
        if not os.path.exists(self.path_file):
            self.read_paths()

        with open(self.path_file,"r") as f:
            paths = json.load(f)
        paths = check_year(paths, years)
        self.paths = paths
    

    def load_done(self):
        if not os.path.exists(self.done_file):
            self.done = {} # dictionary to fast check if the file is downloaded
        else:
            with open(self.done_file,"r") as f:
                self.done = json.load(f)

    def get_main_cc(self):
        command = f"aws s3 ls s3://commoncrawl/crawl-data/ >{self.main_cc_path}"
        os.system(command)
        time.sleep(0.1)

    def read_cc_from_files(self):
        with open(self.main_cc_path,"r") as f:
            cc = f.readlines()
        data = []
        for c in cc:
            split = c.split()
            if "MAIN" in split[1]:
                data.append(split[1].replace("/",""))
        return data
    
    def get_segments(self,cc):
        command = f"aws s3 ls s3://commoncrawl/crawl-data/{cc}/segments/ >{self.segment_path}"
        os.system(command)
        time.sleep(0.01)
        with open(self.segment_path,"r") as f:
            segments = f.readlines()
        return_ = []
        for s in segments:
            split = s.split()
            return_.append(split[1].replace("/",""))
        return return_
    

    def get_files(self,cc, segment):
        command = f"aws s3 ls s3://commoncrawl/crawl-data/{cc}/segments/{segment}/{self.cc_type}/ >{self.file_path}"
        os.system(command)
        # time.sleep(0.1)
        with open(self.file_path,"r") as f:
            files = f.readlines()
        return_ = []
        for f in files:
            split = f.split()
            return_.append(split[3])
        return return_
    
    def read_paths(self):
        self.get_main_cc()
        data = []
        full_path = "s3://commoncrawl/crawl-data/{}/segments/{}/{}/{}"
        cc = self.read_cc_from_files()
        for c in cc:
            segments = self.get_segments(c)
            for s in segments:
                files = self.get_files(c, s)
                for f in files:
                    remote_path = full_path.format(c, s, self.cc_type, f)
                    file_name = f
                    data.append({"cc":c,"segment":s,"file":f,"remote_path":remote_path,"file_name":file_name})
                if self.test_mode:
                    break
            if self.test_mode and len(data) > 100:
                break  
        with open(self.path_file, "w") as f:
            json.dump(data,f,indent=4)
    


    def run(self, years:list= []):
        print("loading paths")
        self.load_paths(years)
        print("loading done")
        self.load_done()
        if self.cc_type == "wet":
            _files = os.listdir(self.download_dir)
            files = [f_ for f_ in _files if f_.endswith(".warc.wet.gz")]
        else:
            if self.cc_type == "warc":
                _files = os.listdir(self.download_dir)
                files = [f_ for f_ in _files if f_.endswith(".warc.gz")]

        print(f"There are {len(files)} to read")
        if self.num_workers == 1:
            for file_path in self.download():
                self.analyze(file_path)
        else:
            if self.num_workers > 1:
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    for f in files:
                        file_path = os.path.join(self.download_dir,f)
                        executor.submit(self.analyze, file_path)
                    for file_path in self.download():            
                        executor.submit(self.analyze, file_path)
            else:
                for file_path in self.download():
                    self.analyze(file_path)



    def check_downloadable(self):
        """Check if the number of files in the output directory is less than the download limit"""
        files = os.listdir(self.download_dir)
        if len(files) < self.download_limit:
            return True
        return False
    
    def download(self):
        count = 0
        for path in self.paths:
            while not self.check_downloadable():
                sleep(1)
            count += 1
            if path["remote_path"] in self.done:
                    continue
            else:
                self.done[path["remote_path"]] = 1
                if count % 10 == 0:
                    with open(self.done_file,"w") as f:
                        json.dump(self.done, f, indent=4)
            remote = path["remote_path"]
            file_name = path["file_name"]
            local = os.path.join(self.download_dir, file_name)
            command = f"aws s3 cp {remote} {local}"
            file_path = local
            while(not os.path.isfile(file_path)):
                os.system(command)
                time.sleep(0.1)
            yield file_path

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
        with open(unzip_path, "rb") as stream:
            for record in ArchiveIterator(stream):
                if record.rec_type == 'response':
                    target_uri = record.rec_headers.get_header('WARC-Target-URI')
                    raw_text = record.content_stream().read()
                    text = extract_text_from_html(raw_text)
                    if text == "":
                        continue
                    _data = {"text":text, "url":target_uri,"id_": BaseCollector.id_generator("_WARC")}
                    rs = self.apply_filters(_data)
                    if rs:
                        data.append(_data)
        print(f"done analyze {input_path}")
        BaseCollector.save(data, data_file)
        os.remove(input_path)#delete the original file
        os.remove(unzip_path)#delete the unzipped file
        print(f"done analyze and delete {input_path} \n unzip {unzip_path} \n save {data_file}")





    