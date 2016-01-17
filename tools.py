import os
import numpy as np
from settings import Settings

src_dir = Settings.DATA_DIR
dst_dir = Settings.DATA_DIR

categories = Settings.CATEGORIES
features = Settings.FEATURES

for categ in categories:
  for ftype in ['TOPICS_64']:
    filename = '%s_%s_features' % (categ, ftype)
    a = np.load(os.path.join(src_dir, '%s.npy' % filename))
    print filename, a.shape
    np.savetxt(os.path.join(dst_dir, '%s.txt' % filename), a)