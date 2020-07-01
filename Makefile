.PHONY: data

data: data/dummy_matlab.mat data/test_scores.h5 data/test_index.yaml data/config.yaml data/test_model.p
data/dummy_matlab.mat:
	aws s3 cp s3://moseq2-testdata/model data/ --request-payer=requester --recursive

data/test_scores.h5:
	aws s3 cp s3://moseq2-testdata/pca data/ --request-payer=requester --recursive

data/test_index.yaml:
	aws s3 cp s3://moseq2-testdata/pca data/ --request-payer=requester --recursive

data/config.yaml:
	aws s3 cp s3://moseq2-testdata/pca data/ --request-payer=requester --recursive

data/test_model.p:
	aws s3 cp s3://moseq2-testdata/model data/ --request-payer=requester --recursive