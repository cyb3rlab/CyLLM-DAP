from base.base_collector import BaseCollector
import requests
import os
import json
import time
import gzip
import shutil
import concurrent.futures
import requests

class S2OCR_Collector(BaseCollector):
    def __init__(self,name:str ="", base_dir:str="", api_key:str ="", test_mode:bool= False,num_workers:int =1, download_limit:int =5):
        super().__init__(name, des="S2OCR Collector")
        self.base_dir = base_dir
        self.num_workers = num_workers
        self.test_mode = test_mode
        self.raw_dir = os.path.join(self.base_dir,"raw")
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        self.api_search_dir = os.path.join(self.base_dir,"api_search")
        if not os.path.exists(self.api_search_dir):
            os.makedirs(self.api_search_dir)
        self.temp_dir = os.path.join(self.base_dir,"temp")
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        self.download_dir = os.path.join(self.base_dir,"download")
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
        self.headers = {'x-api-key': api_key}
        self.done_file = os.path.join( self.base_dir,"done_file.json")
        self.output_dir = os.path.join(self.base_dir,"output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.download_limit = download_limit
    def get_related_paper(self, keyword, fields="title,paperId,externalIds"):
        query = keyword.replace(" ","%20")
        results = []
        keyword = keyword.replace(" ","_")
        path = os.path.join(self.api_search_dir,keyword+ ".json")
        if os.path.exists(path):
            return
        url = f"http://api.semanticscholar.org/graph/v1/paper/search/bulk?query={query}&fields={fields}&year=1980-"
        r = requests.get(url,headers=self.headers).json()
        continued_token = None
        if 'data' in r:
            results.extend(r['data'])
            continued_token = r['token']
        if continued_token:
            while continued_token:
                time.sleep(1)
                url = f"http://api.semanticscholar.org/graph/v1/paper/search/bulk?query={query}&fields={fields}&year=1980-&token={continued_token}"
                r = requests.get(url,headers=self.headers).json()
                if 'data' in r:
                    results.extend(r['data'])
                if 'token' in r and not self.test_mode:
                    continued_token = r['token']
                else:
                    continued_token = None
            
        with open(path,"w") as file:
            json.dump(results,file,indent=4)
    
    def start_bulk_search(self, keywords = []):
        for keyword in keywords:
            keyword = keyword.strip()
                # keyword = keyword.lower()
            print(f"staring retrieving {keyword}")
            self.get_related_paper(keyword)
            print(f"finish retrieving {keyword}")
            if self.test_mode:
                break

    def load_done(self):
        if not os.path.exists(self.done_file):
            self.done = []
        else:
            with open(self.done_file,"r") as f:
                self.done = json.load(f)

    def run(self):
        self.load_done()
        if self.num_workers == 1:
            for f in self.download():
                self.analyze(f)
                if self.test_mode and len(os.listdir(self.output_dir)) > 5:
                    break
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers = self.num_workers) as executor:  
                for f in self.download():
                    executor.submit(self.analyze,f)
                    if self.test_mode and len(os.listdir(self.output_dir)) > 5:
                        break
    def check_downloadable(self):
        """Check if the number of files in the temp directory is less than the download limit"""
        files = os.listdir(self.download_dir)
        if len(files) < self.download_limit:
            return True
        return False
    

    def download(self):
        data = requests.get(f'https://api.semanticscholar.org/datasets/v1/release/latest/dataset/s2orc', headers=self.headers).json()
        files = data["files"]
        count = 0
        for f in files:
            count += 1
            splits = f.split("?AWSAccessKeyId")
            file_name = splits[0].split("/")[-1]
            file_path = os.path.join(self.download_dir, file_name)
            if file_path in self.done:
                continue
            else:
                self.done[file_path] = 1 #save in dictionary for faster lookup
                if count % 100 == 0:
                    self.save_done()
            if os.path.exists(file_path):
                continue
            r = requests.get(f, allow_redirects=True)
            open(file_path, 'wb').write(r.content)
            print(f"finish downloading filepath {file_path}")
            yield file_path                    


    def analyze(self, input_path:str = ""):
        input_name = input_path.split("/")[-1]
        file_path = input_path
        if input_name.endswith(".gz"):
            unzip_name = input_name.replace(".gz",".jsonl")
            unzip_path = os.path.join(self.temp_dir,unzip_name)
            with gzip.open(input_path, 'rb') as f_in:
                with open(unzip_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            file_path = unzip_path
        collected_data = []

        with open(file_path) as file:
            for line in file:
                returned = self.analyze_object(line)     
                if returned:
                    collected_data.append(returned)

        file_id = file_path.split("/")[-1].split(".")[0]
        saved_file_name =  file_id+".jsonl"
        path = os.path.join(self.output_dir, saved_file_name)
        # print(f"start saving {path}")
        BaseCollector.save(collected_data, path)
        os.remove(input_path)
        os.remove(unzip_path)
    
    
    def analyze_object(self, doc ):
        if isinstance(doc, str):
            doc = json.loads(doc)
        text = doc['content']['text']
        annotations = {k: json.loads(v) for k, v in doc['content']['annotations'].items() if v}
        def text_of(type):
            type_indexes = annotations.get(type, '')
            rs = []
            for a in type_indexes:
                start = int(a['start'])
                end = int(a['end'])

                slice = text[start:end]
                rs.append(slice)
            return rs
        external_ids = doc['externalids']
        if not external_ids:
            return None
        paragraphs = text_of('paragraph')
        abstract = text_of('abstract')
        title = text_of('title')
        sections = text_of('sectionheader')
        section_annotations = annotations.get('sectionheader', [])
        paragraph_annotations = annotations.get('paragraph', [])
        new_sections = []
        for i in range(len(section_annotations)):
            current_section_end = int(section_annotations[i]['end'])
            next_section_start = int(section_annotations[i+1]['start']) if i+1 < len(section_annotations) else len(text)
            section_text = sections[i]
            for j in range(len(paragraph_annotations)):
                if int(paragraph_annotations[j]['start']) > current_section_end and int(paragraph_annotations[j]['end']) < next_section_start:
                    section_text += "\n" + paragraphs[j]
            new_sections.append(section_text)
        title = list(set(title))
        external_ids["corpusid"] = doc['corpusid'] if "corpusid" in doc else None
        if len(new_sections) == 0:
            new_sections = paragraphs
        title_text = "\n".join(title)
        full_text = "\n".join(new_sections)
        abstract_text = "\n".join(abstract)
        internal_data = {"title":title_text,"abstract":abstract_text,"full_text":full_text,"external_ids":external_ids, "id_": BaseCollector.id_generator("_S2OCR")}    
        if self.apply_filters(internal_data):
            return internal_data


    

