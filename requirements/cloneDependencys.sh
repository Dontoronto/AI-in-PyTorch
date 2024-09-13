echo "Creating the dependency files in current folder"

#useless i think
#conda list --export > root-env.txt
conda list --export  > root-env.txt


#conda env export --no-build --ignore-channels > environment.yml
conda env export --override-channels -c defaults --no-builds > environment.yml
pip freeze | grep -vE "(git+|http+|file+)" > requirements.txt