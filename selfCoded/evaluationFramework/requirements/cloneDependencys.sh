echo "Creating the dependency files in current folder"

conda env export > environment.yml
pip freeze | grep -vE "(git+|http+|file+)" > requirements.txt