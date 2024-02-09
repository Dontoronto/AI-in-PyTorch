import json

def testmethod(useless="string",path="empty"):
    print(path)

with open('jsonLoadingTest.json', 'r') as json_file:
    args = json.load(json_file)
    print(args)
    #testmethod(**args['transform']['transformParam']['crop_size'])
    testmethod(**args['path'])

    data = args['transform']['transformParam']

    if data['crop_size_Enabled'] == True:
        crop_size = data['crop_size']
    if data['resize_size_Enabled'] == True:
        resize_size = data['resize_size']
    if data['normalize_Enabled'] == True:
        if data['normalize']['mean_Enabled'] == True:
            mean = data['normalize']['mean']
    if data['normalize_Enabled'] == True:
        if data['normalize']['std_Enabled'] == True:
            std = data['normalize']['std']
    if data['interpolation_Enabled'] == True:
        interpolation = data['interpolation']
    if data['antialias_Enabled'] == True:
        antialias = data['antialias']


    print(crop_size)
    print(resize_size)
    print(mean)
    print(type(mean))
    print(std)
    print(interpolation)
    print(antialias)
    # for key, value in data.items():
    #     globals()[key] = value
    #
    # print(crop_size)
    # print(normalize)
