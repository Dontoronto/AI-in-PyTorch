# /models:
This Folder is for saving alle custom made model scripts.

# /models/<subfolder-name>
This child-folder is for saving states of the trained models.

### Notes:
The defaultTrainer is able to export two different model types.
The first one for example 'LeNet.pth' contains the model, optimizer,
current loss and current epoch. You have to like into the code to see
how to import this kind of parameters. The 'raw_LeNet.pth' can be
recognized by its prefix 'raw_' and is just the model. You can 
easily load it via pytorch methods(no workaround).

