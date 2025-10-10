import os
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

NUMS = "1234567890"
RUS_UPPER = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
RUS_LOWER = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
SYMBOLS = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
ALPHABET = NUMS + RUS_UPPER + RUS_LOWER + SYMBOLS
char_to_idx = {c:i for i,c in enumerate(ALPHABET)}
idx_to_char = {i:c for c,i in char_to_idx.items()}

def load_image_and_label(path, size=(128,32)):
    img = Image.open(path).convert("L").resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    label = img.info.get("label", "")
    return np.expand_dims(arr, -1), label

def text_to_labels(text):
    return [char_to_idx[c] for c in text]

image_dir = "dataset"
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(".png")]

X_list, Y_list = [], []
for path in image_paths:
    arr, label = load_image_and_label(path)
    X_list.append(arr)
    Y_list.append(text_to_labels(label))

X = np.stack(X_list)
max_len = max(len(y) for y in Y_list)
Y_padded = keras.preprocessing.sequence.pad_sequences(Y_list, maxlen=max_len, padding="post", value=-1)

input_img = keras.Input(shape=(32,128,1), name="image")
x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(input_img)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2,2))(x)

new_shape = (x.shape[2], x.shape[1]*x.shape[3])
x = layers.Reshape(target_shape=new_shape)(x)

x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Dense(len(ALPHABET)+1, activation="softmax")(x)

model = keras.Model(inputs=input_img, outputs=x)

labels = keras.Input(name="label", shape=(max_len,), dtype="int32")
input_length = keras.Input(name="input_length", shape=(1,), dtype="int32")
label_length = keras.Input(name="label_length", shape=(1,), dtype="int32")

loss_out = layers.Lambda(
    lambda args: keras.backend.ctc_batch_cost(*args),
    output_shape=(1,),
    name="ctc_loss"
)([labels, x, input_length, label_length])

train_model = keras.Model(inputs=[input_img, labels, input_length, label_length], outputs=loss_out)
train_model.compile(optimizer="adam", loss=lambda y_true,y_pred: y_pred)

input_len = np.ones((len(X),1)) * x.shape[1]
label_len = np.array([[len(y)] for y in Y_list])

train_model.fit(
    x=[X, Y_padded, input_len, label_len],
    y=np.zeros(len(X)),
    batch_size=50,
    epochs=10
)

model.save("model.h5")
