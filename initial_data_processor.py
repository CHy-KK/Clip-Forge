import json
import csv
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA 
import logging
data = []
# with open ('initial_text_query.json', 'r') as fj:
#   init_dict = json.load(fj)
#   for key_text, val_emb_list in init_dict.items():
#     for i in range(len(val_emb_list)):
#       data.append([key_text + ' ' + str(i), val_emb_list[i]])
#       # print(len(val_emb_list[i]))

# with open('init_data.csv', 'w', newline='') as fc:
#   writer = csv.writer(fc)
#   writer.writerows(data)
pca = PCA(n_components=2)


shape_embs_list = np.empty(shape=[0,128],dtype=float)
shape_embs = []
with open ('init_data.csv', 'r') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        # row: [str: textquery, list: embedding]
        shape_embs.append([row[0]])
        # print (np.array(row[1][1:-1].split(', '), ndmin=2).astype(np.float))
        shape_embs_list = np.append(shape_embs_list, np.array(row[1][1:-1].split(', '), ndmin=2).astype(np.float), axis=0)
        # shape_embs_torch.append(row[1].type(torch.FloatTensor).to(args.device))
        print(i)
        i += 1
    print (len(shape_embs))
    print (len(shape_embs_list))
    # print (len(shape_embs_torch))
    reduced_shape_embs = pca.fit_transform(shape_embs_list).tolist()
    for i in range(len(shape_embs_list)):
        shape_embs[i].append(reduced_shape_embs[i])

# print(shape_embs)