# main.py

from adapterTest import DictionaryAdapter

class Person:
    def __init__(self, args_dict):
        # Use the adapter to set instance variables based on the dictionary
        self.name = None
        self.age = None
        self.occupation = None
        DictionaryAdapter(self, args_dict)

        print(self.age)   # Output: 28
        print(self.occupation)  # Output: Data Scientist
        print(self.name)  # Output: Jane Doe

def main():
    # Usage example
    args_dict = {'name': 'Jane Doe', 'age': 28, 'occupation': 'Data Scientist'}
    person = Person(args_dict)

    # print(person.name)  # Output: Jane Doe
    # print(person.age)   # Output: 28
    # print(person.occupation)  # Output: Data Scientist

if __name__ == '__main__':
    main()