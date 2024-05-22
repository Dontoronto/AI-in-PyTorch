echo "Creating the dependency files in current folder"
conda list --export > root-env.txt
conda env export > environment.yml
pip freeze | grep -vE "(git+|http+|file+)" > requirements.txt