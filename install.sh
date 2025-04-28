conda create -n videoseg python=3.10

conda activate videoseg

pip install -r requirements.txt

# Install Pytorch Correlation
git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git
cd Pytorch-Correlation-extension
python setup.py install
cd -
