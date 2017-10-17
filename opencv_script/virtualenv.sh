export envname=tmp

echo -e "\nSetup virtual environment..." && sleep 3
if [[ $(grep 'virtualenvwrapper' ~/.bashrc) ]]; then
  echo ".bashrc has been updated!"
else  
    sudo pip install virtualenv virtualenvwrapper
    echo -e "\n# virtualenv and virtualenvwrapper" >> ~/.bashrc
    echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc
    echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
    source ~/.bashrc
fi
source `which virtualenvwrapper.sh`
mkvirtualenv $envname -p python2
python -m ipykernel install --user --name=$envname
workon $envname