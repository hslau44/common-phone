
class BaseProcessor(object):
    
    def __init__(self,**kwargs):
        pass
    
    def __call__(self,inputs):
        raise NotImplementedError()
