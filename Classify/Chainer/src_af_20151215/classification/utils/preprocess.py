import numpy as np

def global_contrast_norm(x):
  if not x.dtype == np.float32:
      x = x.astype(np.float32)
  # x = x.transpose((1, 2, 0))
  # local contrast normalization
  for ch in range(x.shape[0]):
      im = x[ch, :, :]
      im = (im - np.mean(im)) / \
          (np.std(im) + np.finfo(np.float32).eps)
      x[ch, :, :] = im
  return x
  # return x.transpose((2, 0, 1))