from os.path import *
from os import *
import os
import numpy as np
import pandas as pd

root = '01-02-21_01-21' ### insert timestamp here
txt_ext = '.txt'
data_splits = [join(root,el) for el in listdir(root) if isdir(join(root,el))]

for data_split in data_splits:
	models = [join(data_split,model) for model in listdir(data_split) if isdir(join(data_split,model))]
	for model in models:
		files = [join(model,file) for file in listdir(model) if txt_ext in file]
		data = []
		for file in files:
			thick_arr = np.loadtxt(file)
			data.append([file[file.rindex('/')+1:file.index('.')]] + thick_arr.tolist())

		df = pd.DataFrame(data)
		df.to_excel(join(data_split,model[model.rindex('/')+1:]+'.xlsx'), header = False, index=False)
		print(model[model.rindex('/')+1:] + ' done')

print('all done')
