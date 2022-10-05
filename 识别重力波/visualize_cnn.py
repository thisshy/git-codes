from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations

from keras.models import load_model

from matplotlib import pyplot as plt

import numpy as np

# np.set_printoptions(threshold=np.inf)

#加载训练好的mymodel模型
model = load_model("./mymodel")

layer_idx = utils.find_layer_idx(model, 'preds')

model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

#filter_idx = x 是可视化最后一层第x个神经元
filter_idx = 0
img = visualize_activation(model, layer_idx, filter_indices=filter_idx)


plt.imshow(img[..., 0])
plt.show()