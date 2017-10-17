
export envname=cv2

#virtualenv must be created before.

sudo apt-get update
sudo apt-get upgrade
# sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
# sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
# sudo apt-get install libxvidcore-dev libx264-dev
# sudo apt-get install libgtk-3-dev
# sudo apt-get install libatlas-base-dev gfortran

echo -e "\nUpdate and Upgrade system..." && sleep 3
sudo apt update -y && sudo apt upgrade -y

echo -e "\nInstall developer tools..." && sleep 3
sudo apt install -y build-essential cmake pkg-config

echo -e "\nInstall tools for image process..." && sleep 3
sudo apt install -y libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev

echo -e "\nInstall tools for video process..." && sleep 3
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev

echo -e "\nInstall tools for GUI & optimization..." && sleep 3
sudo apt install -y libgtk-3-dev libatlas-base-dev gfortran

echo -e "\nInstall python development library & pip..." && sleep 3
sudo apt install -y python2.7-dev python-pip

sudo apt-get install python2.7-dev python3.5-dev
sudo apt-get install cmake

echo "\nDownloading opencv & opencv_contrib version:..." && sleep 3
cd && wget -O opencv.zip https://github.com/opencv/opencv/archive/3.2.0.zip && unzip opencv.zip 
cd && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.2.0.zip && unzip opencv_contrib.zip 

source `which virtualenvwrapper.sh`
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
mkvirtualenv $envname -p python2
workon $envname


echo -e "\nInstall opencv ..." && sleep 3
pip install numpy
cd ~/opencv-3.2.0/
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.2.0/modules \
    -D PYTHON_EXECUTABLE=~/.virtualenvs/$envname/bin/python \
    -D BUILD_EXAMPLES=ON ..
   
sudo make -j4
sudo make install



echo '*************************************************'
ls -l /usr/local/lib/python2.7/site-packages/
echo '*************************************************'


cd ~/.virtualenvs/$envname/lib/python2.7/site-packages/
sudo ln -s /usr/local/lib/python2.7/site-packages/cv2.so cv2.so