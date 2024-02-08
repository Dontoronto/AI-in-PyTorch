import json

def testmethod(useless="string",path="empty"):
    print(path)

with open('jsonLoadingTest.json', 'r') as json_file:
    args = json.load(json_file)
    testmethod(**args['path'])
