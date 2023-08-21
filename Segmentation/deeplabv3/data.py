import dataloader

data=dataloader.Data_load()


train_data=data.data_load('./data/','voc',{ 'year': '2012','image_set': 'train','download': True})