# deepLearning-tf-keras

## 1. Load mobilenet model with error like "module 'keras_applications.mobilenet' has no attribute 'relu6'"
``` for keres version 2.2.4
with CustomObjectScope({
    'relu6':keras.layers.ReLU(6.),
    'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
    model = load_model(weights_path)
```

## 2. output node name in tf/keras

### check output node name from a keras model
```
node_name = [node.op.name for node in model.outputs]
print(node_name)
```

### check node name in tf (after the graph is defined)
```
node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
print(node_names)
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

## 4. Keras model to tflite: though not working
```
import tensorflow as tf
working_path='/tmp/tflite'
tf.keras.backend.set_learning_phase(1)
inputs = tf.keras.Input(shape=(224, 224, 3), name='input')
y_true = tf.keras.Input(shape=[1000], name='label')
y_pred = keras_model.output

loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)

# quant aware training
graph = tf.get_default_graph()
tf.contrib.quantize.create_training_graph(input_graph=graph, quant_delay=0)

train_step = tf.train.GradientDescentOptimizer(learning_rate=0.00625).minimize(loss)
saver = tf.train.Saver()
with tf.Session() as sess:
    _input = np.random.rand(10, 224, 224, 3)
    _label = np.zeros([10, 1000])
    _label[:, 2] = np.ones(10)
    sess.run(tf.global_variables_initializer())
    for i in range(3):
        _, _loss = sess.run([train_step, loss], feed_dict={inputs: _input,
                                                           y_true: _label})
        print(_loss)
    # save
    saver.save(sess, os.path.join(working_path, 'checkpoints/model.ckpt'))

'''load and convert to TFLite'''
tf.reset_default_graph()
tf.keras.backend.set_learning_phase(0)
inputs = tf.keras.Input(shape=(224, 224, 3), name='input')
keras_model = tf.keras.applications.MobileNetV2(input_tensor=inputs, alpha=1.0, weights=None, include_top=True)
output = keras_model.output

# insert fake quant nodes
graph = tf.get_default_graph()
tf.contrib.quantize.create_eval_graph(graph)
saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(os.path.join(working_path, 'checkpoints/')))

    # freeze graph
    graph_def = graph.as_graph_def()
    froze_graph = tf.graph_util.convert_variables_to_constants(sess, graph_def, [output.op.name])
    tf.io.write_graph(froze_graph, working_path, 'freeze_graph.pb')

# convert to TFLite
graph_def_file = os.path.join(working_path, 'freeze_graph.pb')
input_array = ["input"]
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_array, [output.op.name],
                                                      input_shapes={"input": [1, 224, 224, 3]})

converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
converter.inference_input_type = tf.lite.constants.QUANTIZED_UINT8
converter.quantized_input_stats = {"input": (0., 255.)}
tfmodel = converter.convert()
open(os.path.join(working_path, "converted_model.tflite"), "wb").write(tfmodel)
```
## 4. Manipulate stock keras model
```
def maik_model(img_input):
    """
    :param img_input: instance of layers.Input()
    :return: a modified version of MobileNetV2 model
    """
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=img_input,
        pooling='avg')

    for layer in base_model.layers:
        layer.trainable = True  # trainable has to be false in order to freeze the layers

    base_model.layers.pop()

    op = Conv2D(512, (3, 3), strides=(1, 2), padding='same', activation='relu')(base_model.layers[-1].output)
    op = MaxPooling2D(pool_size=(2, 2))(op)
    op = Conv2D(256, (1, 1), padding='same', activation='relu')(op)
    output_tensor = GlobalMaxPooling2D()(op)

    model = Model(inputs=img_input, outputs=output_tensor, name='maik_model')
    return model
```
