data: data/dummy_matlab.mat
data/dummy_matlab.mat:
	aws s3 cp s3://moseq2-testdata/model data/ --request-payer=requester --recursive