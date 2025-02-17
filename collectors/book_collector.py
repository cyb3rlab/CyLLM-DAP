
import libgen_scraper as lg
import pandas as pd
from urllib.request import urlretrieve
import os
from time import sleep
import concurrent.futures
import json

from utils.pdf2text import pdf2text

from base.base_collector import BaseCollector

def pdf_to_text(input_file:str = "", output_file:str = ""):
        rs = pdf2text(input_file)
        rs["id_"] = BaseCollector.id_generator("_BOOK")
        if rs:
            with open(output_file, "w") as f:
                json.dump(rs, f)
                
class Book_Collector(BaseCollector):
    def __init__(self, name:str ="", base_dir:str = "", test_mode:bool = True, num_workers:int = 5, search_config:dict = {}, keyword_file:str = ""):
        super().__init__(name, des="Book Collector")
        self.base_dir = base_dir
        self.test_mode = test_mode
        self.num_workers = num_workers
        self.output_dir = os.path.join(self.base_dir,"output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.download_dir = os.path.join(self.base_dir,"download")
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
        self.temp_dir = os.path.join(self.base_dir,"temp")
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        self.unzip_dir = os.path.join(self.base_dir,"unzip")
        if not os.path.exists(self.unzip_dir):
            os.makedirs(self.unzip_dir)
        self.search_config = search_config
        self.done_file = os.path.join(self.base_dir,"done_file.json")
        self.keyword_file = keyword_file

    def load_keywords(self):
        if not os.path.exists(self.keyword_file):
            return []
        with open(self.keyword_file, "r") as f:
            self.keywords = f.readlines()
            self.keywords = [k.strip() for k in self.keywords]

    def load_done(self):
        if not os.path.exists(self.done_file):
            self.done = []
        else:
            with open(self.done_file, "r") as f:
                self.done = json.load(f)

    def run(self):
        for data in self.download():
            if data["file_path"]:
                file_name = data["file_path"].split("/")[-1].split(".")[0]
                pdf_to_text(data["file_path"], file_name)

    def link_retriever(self):
        language = self.search_config["language"]
        extension = self.search_config["extension"]
        search_limit = self.search_config["search_limit"]
        
        for k in self.keywords:
            non_fiction = lg.search_non_fiction(
                k,
                search_in_fields=lg.NonFictionSearchField.PUBLISHER,
                filter={
                    lg.NonFictionColumns.LANGUAGE: language,
                    lg.NonFictionColumns.EXTENSION: extension,
                },
                limit= search_limit,
                # chunk_callback=lambda results: print(results.title(0)),
                libgen_mirror="http://libgen.rs",
            )
            result = len(non_fiction)
            if result == 0:
                continue
            for r in range(result):
                data = {}
                data["title"] = non_fiction.title(r)
                data["authors"] = non_fiction.authors(r)
                data["year"] = non_fiction.year(r)
                data["publisher"] = non_fiction.publisher(r)
                data["download_links"] = non_fiction.download_links(r)
                yield data


    def book_download(self, data):
        title = data["title"]
        year = data["year"]
        download_links = data["download_links"]
        filename = title + "_" + str(year) + ".pdf"
        file_path = os.path.join(self.working_dir, filename)
        try:
            print(f"Downloading {title}")
            urlretrieve(download_links[0], file_path)
        except:
            try:
                urlretrieve(download_links[1], file_path)
            except:
                print(f"Download failed for {title}")
                data["file_path"] = None
                return data
        data["file_path"] = file_path
        return data
    def download(self):
        self.load_keywords()
        self.load_done()
        data = self.link_retriever()

        for d in data:
                file_path  = self.book_download(d)
                yield file_path
        # else:
        #     for d in data:
        #         sleep(1)
        #         with concurrent.futures.ProcessPoolExecutor(max_workers= self.num_workers) as executor:
        #             futures.append(executor.submit(self.book_downloader, d))
        
        #     for future in concurrent.futures.as_completed(futures):
        #         yield future.result()







