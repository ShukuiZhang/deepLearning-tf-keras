# deepLearning-tf-keras

## 1. Load mobilenet model with error like "module 'keras_applications.mobilenet' has no attribute 'relu6'", print output node name
``` for keres version 2.2.4
with CustomObjectScope({
    'relu6':keras.layers.ReLU(6.),
    'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
    model = load_model(weights_path)
    
# check output node name
node_name = [node.op.name for node in model.outputs]
print(node_name)

```
