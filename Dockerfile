FROM continuumio/anaconda

# yes I have tried miniconda, I get a segfault that I simply cannot get rid of

# all credit to @alexbw for most of this
# Make sure we can see everything in conda before local install
ENV PATH /opt/conda/lib:/opt/conda/include:$PATH
RUN DEBIAN_FRONTEND=noninteractive apt-get update -y

# Get a newer build toolchain
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential && apt-get install -y lsb-release

# Build a little home for our code
ENV SRC /src
RUN mkdir -p $SRC

# Get our Python requirements (needed to build pyhsmm and pybasicbayes)
RUN conda install pip mpi4py click matplotlib gcc cython -y

# Install moseq
COPY . $SRC/kinect_modeling
RUN pip install -e $SRC/kinect_modeling

# fix bug in Conda matplotlib implementation
# https://github.com/ContinuumIO/anaconda-issues/issues/1068
RUN conda install pyqt=4.11
