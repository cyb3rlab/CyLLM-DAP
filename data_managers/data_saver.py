import json
import pandas as pd
import jsonlines
class DataSaver:
    def __init__(self, data_dir:str = "", file_type:str = "json"):
        self.output_dir = data_dir
        self.file_type = file_type
    def save(self, data, file_name:str = ""):
        if file_name is None:
            print()
        if self.file_type ==  "any":
            self.file_type = "jsonl"
        
        if self.file_type == "json":
            self.save_json(data, file_name)
        if self.file_type == "csv":
            self.save_csv(data, file_name)
        if self.file_type == "jsonl":
            self.save_jsonl(data, file_name)



    def save_json(self, data: object, file_name: str):
        if data is None:
            return
        file_path = f"{self.output_dir}/{file_name}.json"
        with open(file_path, 'w') as f:
            json.dump(data, f)
    

    def save_csv(self, data: object, file_name: str):
        file_path = f"{self.output_dir}/{file_name}.csv"

        data.to_csv(file_path, index=False)

    
    def save_jsonl(self, data: object, file_name: str):
        file_path = f"{self.output_dir}/{file_name}.jsonl"

        with jsonlines.open(file_path, 'w') as writer:
            writer.write_all(data)





