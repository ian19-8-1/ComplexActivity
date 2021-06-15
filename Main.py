from Datasets import BreakfastTexts, BreakfastClips

import matplotlib.pyplot as plt
import numpy as np
# import torch
from matplotlib import cm
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.datasets import MNIST
# from visualization import ANN

from scipy.spatial.distance import cosine


bf_texts = BreakfastTexts('embeddings.pkl')
print('Shape:', len(bf_texts), 'x', len(bf_texts[0]))

# bf_clips = BreakfastClips()
# print('Shape:', len(bf_clips), 'x', len(bf_clips[0]))
# sample = bf_clips.get_sample(6, 5)
# print('Sample:', len(sample), 'x', len(sample[0]))

numpy_texts = []
for i in range(len(bf_texts)):
    numpy_texts.append(bf_texts[i].detach().numpy())

tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(numpy_texts)
# Plot those points as a scatter plot and label them based on the pred labels
cmap = cm.get_cmap('tab20')
fig, ax = plt.subplots(figsize=(8,8))

verbs = ['silence',
         'add',
         'butter',
         'crack',
         'cut',
         'fry',
         'peel',
         'pour',
         'put',
         'smear',
         'spoon',
         'squeeze',
         'stir',
         'stirfry',
         'take']
indices = [[0],
           [1, 2],
           [3],
           [4],
           [5,6,7],
           [8,9],
           [10],
           [11,12,13,14,15,16,17,18,19,20],
           [21,22,23,24,25],
           [26],
           [27,28,29],
           [30],
           [31,32,33,34,35,36,37,38],
           [39,40,41,42,43,44,45,46,47]]

for i in range(len(indices)):
    ax.scatter(tsne_proj[indices[i],0],tsne_proj[indices[i],1], c=np.array(cmap(i)).reshape(1,4), label = verbs[i] ,alpha=0.5)
ax.legend(fontsize='large', markerscale=2)
plt.show()

fry_stirfry = 1 - cosine(numpy_texts[8], numpy_texts[38])
stirfry_crack = 1 - cosine(numpy_texts[38], numpy_texts[4])
crack_fry = 1 - cosine(numpy_texts[4], numpy_texts[8])
print("Fry and stirfry:", fry_stirfry)
print("Stirfry and crack:", stirfry_crack)
print("Crack and fry:", crack_fry)
