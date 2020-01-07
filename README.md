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
## 2. What to do with `.pb` file in tensorflow
pb stands for protobuf. In TensorFlow, the protbuf file contains the graph definition as well as the weights of the model. Thus, a pb file is all you need to be able to run a given trained model.

Given a pb file, you can load it as follow.
```
def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph
```
Once you have loaded the graph, you can basically do anything. For instance, you can retrieve tensors of interest with
```
input = graph.get_tensor_by_name('input:0') # input node name
output = graph.get_tensor_by_name('output:0') # output node name
```
and use regular TensorFlow routine like:
```
sess.run(output, feed_dict={input: some_data})
```
## 3. save checkpoint to .pb
```
import tensorflow as tf

meta_path = 'model.ckpt-22480.meta' # Your .meta file
output_node_names = ['output:0']    # Output nodes

with tf.Session() as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint('path/of/your/.meta/file'))

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('output_graph.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())
```      
