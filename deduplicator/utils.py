


from datasketch import MinHash
from datasketch import MinHashLSH
import pickle
import os
import concurrent.futures
import json
import pathlib
from data_managers.data_loader import DataLoader
from time import sleep
def text_to_nram(text:str, ngram:int = 5):
    text = text.replace("\n"," ")
    text = text.replace("\t"," ")
    text = text.replace("  "," ")
    text = text.strip().lower()
    shingles = text.split(" ")
    return set(["_".join(shingles[i:i+ngram]) for i in range(len(shingles)-ngram+1)])

def text_to_minhash(saved_path:str, text:str, ngram:int = 5, num_perm:int = 128):
    m = MinHash(num_perm=num_perm)
    shingles = text_to_nram(text,ngram)
    for shingle in shingles:
        m.update(shingle.encode('utf8'))
    with open(saved_path,"wb") as f:
        pickle.dump(m,f)
    # print("process finished",flush=True)
    return


def texts_to_minhashes(data_loader:DataLoader = None, saved_dir:str = "", ngram:int = 5, num_perm:int = 128, num_workers:int = 1):
    if num_workers == 1:
        for id_, text_ in data_loader.iter_load():
            saved_path = os.path.join(saved_dir,f"{id_}.pkl")
            text_to_minhash(saved_path,text_,ngram,num_perm)
    else:
        with concurrent.futures.ProcessPoolExecutor(num_workers) as executor:
            count = 0
            for id_, text_ in data_loader.iter_load():
                count += 1
                if count % 1000 == 0:
                    sleep(1)
                saved_path = os.path.join(saved_dir,f"{id_}.pkl")
                executor.submit(text_to_minhash,saved_path,text_,ngram,num_perm)


def lshing_from_disk(hash_path_obj, threshold:float = 0.5, num_perm:int = 128):
    """load local minhashes from disk and insert them into LSH"""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    count = 0
    for f in hash_path_obj.rglob('*.pkl'):
        count+=1
        if count % 50000 == 0:
            print(count)
        key = f.stem
        with open(f,"rb") as f:
            m = pickle.load(f)
            lsh.insert(key, m)
    return lsh


def find_duplicates(lsh:MinHashLSH, hash_path_obj):
    duplicates = []
    count = 0
    already = {}
    for f in hash_path_obj.rglob('*.pkl'):
        count+=1
        if count % 50000 == 0:
            print(count)
        id_ = f.stem
        if id_ in already:
            continue # we skip the duplicate
        with open(f,"rb") as f:
            m = pickle.load(f)
            result = lsh.query(m)
            if len(result) > 1:
                duplicates.append(result)
                for r in result:
                    already[r] = 1
    return duplicates


def deduplicate(data_dict:dict = {}, duplicates:list = []):
    removed_dict ={}
    saved_data = []
    for duplicate in duplicates:
        count+=1
        duplicated_items = []
        for d in duplicate:
            duplicated_item = {}
            duplicated_item["text"] = data_dict[d]["text"]
            duplicated_item["id"] = d
            duplicated_items.append(duplicated_item)    
        duplicated_items = sorted(duplicated_items,key = lambda x: len(x["text"]),reverse=True)# sort by length of text in descending order
        for i in range(1,len(duplicated_items)):
            removed_dict[duplicated_items[i]["id"]]=1 #
    for k,v in data_dict.items():
        if k not in removed_dict:
            saved_data.append(v)
    return saved_data


def deduplicating(data_loader = None, saved_dir:str = "", ngram:int = 5, num_perm:int = 128, num_workers:int = 1, threshold:float = 0.5):
    texts_to_minhashes(data_loader,saved_dir,ngram,num_perm,num_workers)
    hash_path_obj= pathlib.Path(saved_dir)
    lsh = lshing_from_disk(hash_path_obj,threshold,num_perm)
    duplicates = find_duplicates(lsh,hash_path_obj)
    saved_data = deduplicate(data_loader.all_data,duplicates,saved_dir)
    return saved_data