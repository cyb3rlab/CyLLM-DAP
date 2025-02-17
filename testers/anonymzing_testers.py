from manager import Manager
data = {
    "text":"I am a bad person. My name is John Doe. I live in New York. My bank account is 1234567890. My phone number is 123-456-7890. My email is adfsf@gmail.com",
    "url":"https://www.google.com"
}



save_config = {
    "class_name":"DataSaver",
    "params": {
        "data_dir":"workspace/anonymized",
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
anonynmizer_config = {
    "class_name":"Anonymizer",
    "params": {
        "entities": ["PHONE_NUMBER","CREDIT_CARD","EMAIL_ADDRESS", "URL"],
        "language":"en",
        "applied_keys": ["text"],
        "num_workers": 2
    },


}

from data_managers.data_loader import DataLoader
from data_managers.data_saver import DataSaver
def test_anonymizer():
    loader = Manager.init_loader(loader_config)
    saver = Manager.init_saver(save_config)
    anonynmizer_config["params"]["data_loader"] = loader
    anonynmizer_config["params"]["data_saver"] = saver
    anonymizer = Manager.init_anonymizer(anonynmizer_config)
    result = anonymizer.run()
    print(result)
