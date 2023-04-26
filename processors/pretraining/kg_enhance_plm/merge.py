import os
import pickle
from tqdm import tqdm

input_folder = "../pretrain_data/data"
output_folder = "../pretrain_data/data"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

normal_data_path = os.path.join(output_folder, "normal")
if not os.path.exists(normal_data_path):
    os.mkdir(normal_data_path)

large_data_path = os.path.join(output_folder, "large")
if not os.path.exists(large_data_path):
    os.mkdir(large_data_path)

file_list = []
for path, _, filenames in os.walk(input_folder):
    for filename in filenames:
        file_list.append(os.path.join(path, filename))
print("# of files", len(file_list))

data = []
i = 0
num_sample_per_file = 10000
num_sample_per_large_file = 10000

for j in tqdm(range(len(file_list))):
    file = file_list[j]
    if ".txt" in file:
        continue
    if "large" in file:
        continue
    # regular file (may not be of 1k)
    tmp = pickle.load(open(file, "rb"))
    # print("processing file {}. # of samples: {}".format(file, len(tmp)))
    data.extend(tmp)
    if len(data) >= num_sample_per_file:
        with open(os.path.join(normal_data_path, str(i)), "wb") as fout:
            pickle.dump(data[:num_sample_per_file], fout, protocol=4)
            i += 1
        if len(data) > num_sample_per_file:
            data = data[num_sample_per_file:]
        else:
            data = []
print("# rest samples: {}".format(len(data)))

large_data = []
i = 0
for j in tqdm(range(len(file_list))):
    file = file_list[j]
    if "large" in file:
        # large file (may not be of 1k)
        tmp = pickle.load(open(file, "rb"))
        # print("processing file {}. # of samples: {}".format(file, len(tmp)))
        large_data.extend(tmp)
        if len(large_data) >= num_sample_per_large_file:
            with open(os.path.join(large_data_path, str(i)), "wb") as fout:
                pickle.dump(large_data[:num_sample_per_large_file], fout, protocol=4)
                i += 1
            if len(large_data) > num_sample_per_large_file:
                large_data = large_data[num_sample_per_large_file:]
            else:
                large_data = []
print("# rest samples: {}".format(len(large_data)))
