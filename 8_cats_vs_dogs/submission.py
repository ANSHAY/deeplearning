#!usr/bin/python
import pandas as pd
import predict
import glob

def save_submission(result, test_list):
	idx = []
	for i in test_list:
	    idx.append(i[21:-4])

	result = result.reshape(result.shape[0])
	result[result>0.5] = 1
	result[result<0.5] = 0

	submission = {"id": idx, "label": result}

	pd.DataFrame(submission).to_csv("submission.csv", index=False)

if __name__ == '__main__':
	test_dir = 'D:/Datasets/cats_vs_dogs/test1/'
	test_list = glob.glob(test_dir+'*.jpg')
	result = predict.predict(test_list = test_list)
	save_submission(result, test_list)
