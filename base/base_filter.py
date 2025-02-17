class BaseFilter:
    def __init__(self, name:str= "", keys:list = []):
        self.keys = keys
        self.name = name


    def run(self):
        """
        Run the filter
        return True if the filter is applied successfully
        """
        raise NotImplementedError
    

    def load_internal_data(self, file_path:str):
        raise NotImplementedError