This assumes you will be running the model on Kubernetes using Google Kubernetes Engine.  If you are using a custom instance of Kubernetes, you can skip the (GKE only) steps.

### Python packages

Moseq2-model has been tested with Python 3.6.  All requirements can be installed using pip, though it's recommended you create a new conda environment to ensure the dependencies do not conflict with any software you may have currently installed.

```sh
conda create -n moseq2_model python=3.6
source activate moseq2_model
git clone https://github.com/dattalab/moseq2_model.git
cd moseq2_model/
pip install -e . --process-dependency-links
```

### Bash scripts (GKE only)

Note that the installation also requires the [kubectl comand line tool](https://kubernetes.io/docs/tasks/tools/install-kubectl/#install-kubectl-binary-via-curl).  Be sure to install before proceeding.  Included in this repo are two convenience scripts for firing up and tearing down a GKE cluster, `scripts/moseq2_model_kube_up.sh` and `scripts/moseq2_model_kube_down.sh`.  Make symlinks to them in /usr/local/bin/,

```sh
ln -s path_to_moseq2_model/scripts/moseq2_model_kube_up.sh /usr/local/bin/moseq2_model_kube_up
ln -s path_to_moseq2_model/scripts/moseq2_model_kube_down.sh /usr/local/bin/moseq2_model_kube_down
chmod a+x /usr/local/bin/moseq2*
moseq2_model_kube_up --help
moseq2_model_kube_down --help
```

### GCloud components (GKE only)

You will need to create a project using the [Google Cloud Platform console](https://cloud.google.com/resource-manager/docs/creating-managing-projects).  Once that is done, let's authorize gcloud,

```sh
gcloud auth login
```

You will need to login through your browser to authorize gcloud.  Now set the project,

```sh
gcloud config set project [PROJECT_ID]
```

Set your compute zone, which typically is `us-east1-a`

```sh
gcloud config set compute/zone us-east1-a
```

Some typical settings are included in `scripts/moseq2_model_kube_settings.sh`.  If you want them to be loaded by default, on a Mac,

```sh
cat path_to_moseq2_model/scripts/model_kube_settings.sh >> ~/.bash_profile
source ~/.bash_profile
```

On linux,

```sh
cat path_to_moseq2_model/scripts/model_kube_settings.sh >> ~/.bashrc
source ~/.bashrc
```

You always change these settings later.  Now let's try a dry run of firing up a cluster,

```sh
moseq2_model_kube_up -d
moseq2_model_kube_down -d
```

These commands will display what would be issued to the command line if you were not performing a dry run.


### Creating a Kubernetes job specification

Now we can create a yaml Kubernetes job specification that will scan over a set of parameter ranges.
