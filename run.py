# Custom Grid Search 
# for running multiple combinations of parameters using this file

import os 

n_estimators = [100 , 150 , 200]
max_depth = [10 , 15 , 20]

for n in n_estimators:
    for m in max_depth:
        os.system(f"python basic_ml_model.py -n {n} -m {m}")