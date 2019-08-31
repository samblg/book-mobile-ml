import os

import shutilimport random

train= '../data/dog_cat/train/'

dogs=[train + i for i in os.listdir(train) if 'dog'in i]
cats=[train + i for i in os.listdir(train)] if 'cat' in i]

print len(dogs)，len(cats)


target = '../data/dog_cat/arrange/'

random.shuffle(dogs)
random.shuffle(cats)

def ensure_dir(dir_path) :
  if not os.path.exists(dir_path):
  try:
    os.makedirs(dir_path)
  except OSError:
    pass

ensure_dir(target + 'train/dog')
ensure_dir(target + 'train/cat')
ensure_dir(target + 'validation/dog')
ensure_dir(target + 'validation/cat')

for dog_file, cat_file in zip(dogs, cats)[:1000]:
  shutil.copyfile(dog_file, target + 'train/dog/' + os.path.basename (dog_file))
  shutil.copyfile(cat_file, target + 'train/cat/' + os.path.basename (cat_file))

for dog_file, cat_f1le in zip(dogs, cats)[1000:1500]:
  shutil.copyfile(dog_file, target + 'validation/dog/' + os.path.basename(dog_file))
  shutil.copyfile(cat_file, target + 'validation/cat/' + os.path.basename(cat_file))


