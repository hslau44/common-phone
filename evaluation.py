

class Evaluator(object):

    def __init__(self,config=None):
        self.config = config
    
    def evaluate(self,model) -> dict: 
        """Return dict of result"""
        raise NotImplementedError()