#!/bin/bash

# can make preemptible

gcloud container clusters create test-cluster --scopes storage-full --machine-type n1-highcpu-8 --num-nodes 3
gcloud container clusters get-credentials test-cluster
gcloud auth application-default login

# tear down with gcloud container clusters delete test-cluster

gcloud container clusters delete test-cluster
