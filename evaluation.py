from sklearn import metrics

class Evaluator(object):

    def __init__(self,config=None):
        self.config = config
    
    def evaluate(self,model) -> dict: 
        """Return dict of result"""
        raise NotImplementedError()


class CommonPhoneSegmentationEvaluator(Evaluator):
    
    def __init__(self,data_dir,test_locales,**kwargs):
        metadata_fp = os.path.join(data_dir,"metadata.csv")
        metadata = get_metadata(path=metadata_fp,_locale=test_locales,_set='test')
        self.dataset = PhonemeDetailsDataset(metadata,data_dir=data_dir)
        super().__init__(config=config)
    
    def evaluate(self,model) -> dict:
        sample_acc = 0
        for i in range(len(self.dataset)):
            inputs = [self.dataset[i]['input_values']]
            outputs = model(inputs)
            pred = outputs[0]['pred']
            label = inputs['labels']
            sample_acc += metrics.accuracy_score(ref,pred) 
        sample_acc /= len(self.dataset)
        result = {'sample_acc':sample_acc}
        return result