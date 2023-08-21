from torchvision.datasets import VOCSegmentation




class Data_load:
    def __init__(self):
        self.data=None
        self.data_name=None
        self.data_parameter=None
        
    def data_load(self, data_path, data_name, data_parameter=None):
        if data_parameter is None:
            if data_name == 'voc':
                data_parameter = {'root': './', 'year': '2012','image_set': 'train','download': True}  # Default parameters for 'voc'
            elif data_name == 'coco':
                data_parameter = {'root': '/'}  # Default parameters for 'coco'
        
        if data_name == 'voc':
            self.data = VOCSegmentation(data_path, **data_parameter)

    



