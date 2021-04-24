import tensorflow as tf
from tensorflow import keras

import numpy as np

from s3fs.core import S3FileSystem
s3 = S3FileSystem()

import os

########
import smdistributed.dataparallel.tensorflow as sdp
sdp.init()
########

########
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[sdp.local_rank()], 'GPU')
########

bucket = 'zhixue.sagemaker.iad'
train_images = np.load(s3.open('{}/{}'.format(bucket, 'lego-simple-train-images.npy')))
train_labels = np.load(s3.open('{}/{}'.format(bucket, 'lego-simple-train-labels.npy')))
test_images = np.load(s3.open('{}/{}'.format(bucket, 'lego-simple-test-images.npy')))
test_labels = np.load(s3.open('{}/{}'.format(bucket, 'lego-simple-test-labels.npy')))

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).repeat(200).shuffle(10000).batch(128)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(128)
    
class_names = ['2x3 Brick', '2x2 Brick', '1x3 Brick', '2x1 Brick', '1x1 Brick', 
               '2x2 Macaroni', '2x2 Curved End', 'Cog 16 Tooth', '1x2 Handles', '1x2 Grill']

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(48, 48)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

########
optimizer = tf.keras.optimizers.Adam()
#optimizer = tf.optimizers.Adam(0.000125 * sdp.size())
checkpoint_dir = os.environ['SM_MODEL_DIR']
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
########

@tf.function
def train_step(images, labels, first_batch):
    with tf.GradientTape() as tape:
    
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
        
    ########
    # Wrap tf.GradientTape with the library's DistributedGradientTape
    tape = sdp.DistributedGradientTape(tape)
    ########
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    ########
    if first_batch:
        # SageMaker data parallel: Broadcast model and optimizer variables
        sdp.broadcast_variables(model.variables, root_rank=0)
        sdp.broadcast_variables(optimizer.variables(), root_rank=0)
    ########
    
    train_loss(loss)
    train_accuracy(labels, predictions)
    
@tf.function
def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    
EPOCHS = 1

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    batch = 0
    for images, labels in train_ds:
        train_step(images, labels, batch == 0)
        if batch % 100 == 0:
            print(
                f'Batch {batch}, '
                f'Loss: {train_loss.result()}, '
                f'Accuracy: {train_accuracy.result() * 100}'
            )
        batch = batch + 1
    
    for t_images, t_labels in test_ds:
        test_step(t_images, t_labels)
    
    print(
        f'Epoch {epoch}, '
        f'Training Loss: {train_loss.result()}, '
        f'Training Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )

######## 
# SMDataParallel: Save checkpoints only from master node.
if sdp.rank() == 0:
    model.save(os.path.join(checkpoint_dir, '1'))
######## 