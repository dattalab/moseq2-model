#!/bin/bash

# bootstrap a slurm environment with the necesseties

curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$HOME/miniconda3_latest.sh"
chmod +x $HOME/miniconda3_latest.sh
$HOME/miniconda3_latest.sh -b -p $HOME/miniconda3
cat >> ~/.bashrc << END
# add for miniconda install
PATH=$HOME/miniconda3:\$PATH
END
source $HOME/.bashrc
conda create -n "moseq2" python=3.6 -y
source activate moseq2
mkdir $HOME/python_repos
git clone https://github.com/dattalab/moseq2_model.git $HOME/python_repos/moseq2_model
git clone https://github.com/dattalab/moseq2.git $HOME/python_repos/moseq2_model
pip install -e $HOME/python_repos/moseq2_model --process-dependency-links
pip install -e $HOME/python_repos/moseq2
