## Installation of python packages

### Python packages

Moseq2-model has been tested with Python 3.6.  All requirements can be installed using pip, though it's recommended you create a new conda environment to ensure the dependencies do not conflict with any software you may have currently installed.

```sh
# you'll need gcc-c++ to compile some components of pybasicbayes
sudo yum install git gcc-c++
# get the latest miniconda3 and install in a cluster-mounted directory
cd ~/
curl -O 'https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh'
chmod a+x Miniconda3-latest-Linux-x86_64
# make sure to select yes for prepending the Miniconda directory to your PATH
./Miniconda3-latest-Linux-x86_64
# make a local environment to install the code
conda create -n moseq2_model python=3.6
source activate moseq2_model
git clone https://github.com/dattalab/moseq2_model.git
cd moseq2_model/
# now install the repo
pip install -e . --process-dependency-links
```

Verify that everything works,

```sh
moseq2-model --help
```
