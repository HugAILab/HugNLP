import json
import os
import random
base_path = './dataset/cpic/'
if not os.path.exists(os.path.join(base_path, 'merge_many_template')):
    os.makedirs(os.path.join(base_path, 'merge_many_template'))
# merge training data
all_data = list()
for file in [
        'my_opend_train_many_template.json', 'my_ins_train_many_template.json'
]:
    data = json.load(open(base_path + file, encoding='utf8'))
    all_data.extend(data)
    if 'open' not in file:
        all_data.extend(data)
        all_data.extend(data)
random.shuffle(all_data)
with open(os.path.join(base_path, 'merge_many_template/train.json'),
          'w',
          encoding='utf8') as fout:
    json.dump(all_data, fout, indent=2, ensure_ascii=False)

# copy test data
file = 'my_ins_test_many_template.json'
data = json.load(open(base_path + file, encoding='utf8'))
with open(os.path.join(base_path, 'merge_many_template/test.json'),
          'w',
          encoding='utf8') as fout:
    json.dump(data, fout, indent=2, ensure_ascii=False)
