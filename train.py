import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUMS = "1234567890"
RUS_UPPER = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
RUS_LOWER = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
SYMBOLS = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
ALPHABET = NUMS + RUS_UPPER + RUS_LOWER + SYMBOLS
char_to_idx = {c: i for i, c in enumerate(ALPHABET)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

IMG_W, IMG_H = 128, 32

def list_image_paths(folder="dataset"):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".png")]

def read_label_from_png(path):
    img = Image.open(path)
    label = img.info.get("label", "")
    img.close()
    return label

all_paths = list_image_paths("dataset")
if len(all_paths) == 0:
    raise RuntimeError("В папке 'dataset' нет PNG файлов.")
all_labels = [read_label_from_png(p) for p in all_paths]

def filter_to_alphabet(s):
    return "".join([c for c in s if c in char_to_idx])

all_labels = [filter_to_alphabet(s) for s in all_labels]
max_len = max(max(len(s) for s in all_labels), 1)

val_ratio = 0.10
n_total = len(all_paths)
n_val = max(1, int(n_total * val_ratio))
indices = np.arange(n_total)
np.random.shuffle(indices)
val_idx = indices[:n_val]
train_idx = indices[n_val:]

train_paths = [all_paths[i] for i in train_idx]
train_labels = [all_labels[i] for i in train_idx]
val_paths = [all_paths[i] for i in val_idx]
val_labels = [all_labels[i] for i in val_idx]

def text_to_indices(s):
    return np.array([char_to_idx[c] for c in s], dtype=np.int32)

def load_sample_py(path_bytes, label_bytes):
    path = path_bytes.decode("utf-8")
    img = Image.open(path).convert("L").resize((IMG_W, IMG_H))
    arr = np.array(img, dtype=np.float32) / 255.0
    img.close()
    arr = np.expand_dims(arr, axis=-1)
    label_str = label_bytes.decode("utf-8")
    indices = text_to_indices(label_str)
    label_len = np.int32(len(indices))
    if len(indices) < max_len:
        pad = np.full((max_len - len(indices),), -1, dtype=np.int32)
        indices = np.concatenate([indices, pad], axis=0)
    else:
        indices = indices[:max_len]
    return arr, indices, label_len

def tf_load_sample(path, label):
    arr, indices, label_len = tf.numpy_function(load_sample_py, [path, label], [tf.float32, tf.int32, tf.int32])
    arr.set_shape((IMG_H, IMG_W, 1))
    indices.set_shape((max_len,))
    label_len.set_shape(())
    return {"image": arr, "label": indices, "label_len": label_len}

def make_dataset(paths, labels, batch_size=64, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
    ds = ds.map(tf_load_sample, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

batch_size = 64
train_ds = make_dataset(train_paths, train_labels, batch_size=batch_size, shuffle=True)
val_ds = make_dataset(val_paths, val_labels, batch_size=batch_size, shuffle=False)

def conv_block(x, filters, kernel_size=(3,3), pool=(2,2)):
    x = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool)(x)
    return x

image_in = keras.Input(shape=(IMG_H, IMG_W, 1), name="image")
x = conv_block(image_in, 64, (3,3), pool=(2,2))
x = conv_block(x, 128, (3,3), pool=(2,2))
x = conv_block(x, 256, (3,3), pool=(2,1))
Hp, Wp, Cp = x.shape[1], x.shape[2], x.shape[3]
time_steps = Wp
feat_dim = Hp * Cp
x = layers.Reshape((time_steps, feat_dim))(x)
x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
x = layers.Dropout(0.3)(x)
x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
logits = layers.Dense(len(ALPHABET) + 1, activation="softmax")(x)
model = keras.Model(image_in, logits, name="crnn_ctc_logits")

labels_in = keras.Input(name="label", shape=(max_len,), dtype="int32")
input_length_in = keras.Input(name="input_length", shape=(1,), dtype="int32")
label_length_in = keras.Input(name="label_length", shape=(1,), dtype="int32")
ctc_loss = layers.Lambda(lambda args: keras.backend.ctc_batch_cost(*args), name="ctc_loss")(
    [labels_in, logits, input_length_in, label_length_in]
)
train_model = keras.Model(inputs=[image_in, labels_in, input_length_in, label_length_in], outputs=ctc_loss)

opt = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=5.0)
train_model.compile(optimizer=opt, loss=lambda y_true, y_pred: y_pred)

def add_lengths(batch):
    bsz = tf.shape(batch["image"])[0]
    input_len = tf.fill((bsz, 1), tf.cast(time_steps, tf.int32))
    label_len = tf.reshape(batch["label_len"], (-1, 1))
    return ((batch["image"], batch["label"], input_len, label_len), tf.zeros((bsz,)))

def to_training_inputs(ds):
    return ds.map(add_lengths, num_parallel_calls=tf.data.AUTOTUNE)

train_inputs = to_training_inputs(train_ds)
val_inputs = to_training_inputs(val_ds)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
]

epochs = 50
history = train_model.fit(train_inputs, validation_data=val_inputs, epochs=epochs, callbacks=callbacks)

model.save("model.h5")

batch = next(iter(val_ds))
images = batch["image"]
probs = model.predict(images)
input_len_vec = np.full((probs.shape[0],), probs.shape[1], dtype=np.int32)
decoded, _ = keras.backend.ctc_decode(probs, input_length=input_len_vec, greedy=True)
decoded = decoded[0].numpy()

def labels_to_text(seq):
    return "".join(idx_to_char[i] for i in seq if i != -1)

texts = [labels_to_text(seq) for seq in decoded]
print("Примеры предсказаний:", texts[:5])
