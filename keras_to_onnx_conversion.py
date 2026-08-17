import os
from datetime import datetime
from typing import Any

# fix error when importing tf2onnx:
# AttributeError: module 'numpy' has no attribute 'bool'
import numpy as np
# Manually restore the missing aliases expected by tf2onnx
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'object'):
    np.object = object


import tensorflow as tf
from numpy import signedinteger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tensorflow import keras
import tf2onnx
import onnx
import onnxruntime as ort
import numpy as np

import config
from ultrasound_gesture_cnn_classification import load_subject_arrays
from visualizations import save_confusion_matrix_png

# on Studio
# keras_model_filename = "cnn_perp_subject_4_200_epochs_224px_20260804_124025.keras"
# keras_model_filename = "cnn_perp_subject_4_100_epochs_224px_20260808_223847.keras"
# on MacBook Pro
# keras_model_filename = "cnn_perp_subject_4_200_epochs_224px_20260803_222438.keras"
keras_model_filename = "cnn_perp_subject_4_200_epochs_224px_20260815_183827.keras"
keras_model_folder_path = "results/models/cnn/"
keras_model_filepath = os.path.join(keras_model_folder_path, keras_model_filename)

onnx_model_folder_path = "results/models/cnn/onnx/"
onnx_model_filepath = os.path.join(onnx_model_folder_path, keras_model_filename.replace(".keras", ".onnx"))

# configuration
confirm_after_conversion: bool = True
output_onnx_confusion_matrix: bool = False
output_keras_confusion_matrix: bool = False


# Load your Keras 2.13 model
# model = tf.keras.models.load_model(keras_model_filepath)  # or your saved model path

learning_rate = 1e-5
input_shape = (224, 224, 1)
num_classes = 12

print(f"Loading model from: {keras_model_filepath}...")
model: keras.Model = keras.models.load_model(keras_model_filepath, compile=False)
# ensure the model is in inference mode
keras.backend.set_learning_phase(0)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

# 2. Define the input signature
# [None] creates a dynamic batch size so you can pass 1 or many images at once
input_signature = [tf.TensorSpec([None] + list(input_shape), tf.float32, name="input_tensor")]

# 3. Convert to ONNX
# Opset 13+ is highly recommended for modern BatchNormalization handling
# Using opset 15+ provides better compatibility for BatchNormalization
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13
    # opset=15
    # opset=16
)

# 4. Save the file
onnx.save(onnx_model, onnx_model_filepath)

print("Model successfully converted to cnn_model.onnx")

###

# Define input signature (replace dimensions to match your exact input shape)
# Use None for dynamic batch size
# input_signature = [tf.TensorSpec([None] + model.input_shape[1:], tf.float32, name="input")]

# Convert Keras model to ONNX proto
# onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)

# Save to disk
# onnx.save(onnx_model, onnx_model_filepath)

if confirm_after_conversion:
    # compare the predictions of the keras and onnx models for a single image (True) or a batch of images (False)
    test_single = True
    # test_single = False
    test_batch_size = 12
    
    print(f"Comparing Keras and ONNX predictions...")
    # Create dummy input data matching your shape
    dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
    print(f"dummy_input shape: {dummy_input.shape}")
    
    # load example image from test data
    data_filename: str = "X_test.npy"
    data_root_folder_path: str = config.default_dataset_path
    data_mode: str = "perp"
    data_subject: str = "Subject_4"
    data_image_size: int = 224

    (x_train, y_train), (x_test, y_test), data_num_classes = load_subject_arrays(
        data_root_folder_path, data_mode, data_subject, data_image_size
    )
    
    if test_single:
        # get a single test image with shape (224, 224, 1)
        test_image = x_test[0].astype(np.float32)
        # put the test image into a batch of size 1, shape (1, 224, 224, 1)
        test_image = np.expand_dims(test_image, axis=0)
    else:
        # get a batch of test images with shape (N, 224, 224, 1), evenly sampled
        x_test_size: int = len(x_test)
        batch_increment: int = int(x_test_size / test_batch_size)
        print(f"test_batch_size: {test_batch_size}, x_test_size: {x_test_size}, batch_increment: {batch_increment}")
        test_image = x_test[10:x_test_size:batch_increment].astype(np.float32)
    
    print(f"test_image shape: {test_image.shape}")
    
    # Get Keras prediction logits
    # keras_pred = model.predict(test_image)
    keras_pred = model(test_image, training=False).numpy()
    # keras_pred_probs = np.softmax(keras_pred, axis=1)
    # convert to normalized probabilities using softmax
    # keras_pred = keras_pred / np.sum(keras_pred, axis=1, keepdims=True)
    keras_pred_probs = tf.nn.softmax(keras_pred, axis=1)
    
    # softmax is:
    # tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis, keepdims=True) 
    
    print("Keras prediction shape: ", keras_pred.shape)
    print("Keras prediction:", keras_pred)
    print("Keras prediction probs:", keras_pred_probs)
    keras_pred_abs = [np.argmax(x) for x in keras_pred_probs]
    # print("Keras prediction abs:", np.argmax(keras_pred_probs, axis=1))
    print("Keras prediction abs:", keras_pred_abs)
     
    # Get ONNX prediction
    ort_sess = ort.InferenceSession(onnx_model_filepath)
    onnx_pred = ort_sess.run(None, {"input_tensor": test_image})[0]
    # convert to normalized probabilities using softmax
    # onnx_pred = onnx_pred / np.sum(onnx_pred, axis=1, keepdims=True)
    onnx_pred_probs = tf.nn.softmax(onnx_pred, axis=1)
    
    print("ONNX prediction shape: ", onnx_pred.shape)
    print("ONNX prediction:", onnx_pred)
    print("ONNX prediction probs:", onnx_pred_probs)
    onnx_pred_abs = [np.argmax(x) for x in onnx_pred]
    # print("ONNX prediction abs:", np.argmax(onnx_pred, axis=1))
    print("ONNX prediction abs:", onnx_pred_abs)
    
    # compare absolute class predictions
    for i in range(len(keras_pred_abs)):
        preds_match = keras_pred_abs[i] == onnx_pred_abs[i]
        match_char = "✓" if preds_match else "✗"
        print(f"Keras prediction abs vs ONNX prediction abs: {keras_pred_abs[i]} vs {onnx_pred_abs[i]} {match_char}")
    
    # Check if they are identical
    # np.testing.assert_allclose(onnx_pred, keras_pred, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(onnx_pred_probs, keras_pred_probs, rtol=1e-5, atol=1e-5)
    print("ONNX outputs match Keras to within 1e-5")

# add y_pred variables to the global scope so they can be compared
y_pred_onnx: signedinteger[Any] | np.ndarray[Any, np.dtype[Any]] | Any
y_pred_keras: signedinteger[Any] | np.ndarray[Any, np.dtype[Any]] | Any

if output_onnx_confusion_matrix:
    # test_single = True
    test_onnx_single = False
    
    # load example image from test data
    data_root_folder_path: str = config.default_dataset_path
    data_mode: str = "perp"
    data_subject_id: str = "4"
    data_subject: str = f"Subject_{data_subject_id}"
    data_image_size: int = 224

    (x_train, y_train), (x_test, y_test), data_num_classes = load_subject_arrays(
        data_root_folder_path, data_mode, data_subject, data_image_size
    )

    # test_onnx_batch_size = 10
    test_onnx_batch_size: int = len(x_test)

    if test_onnx_single:
        # get a single test image with shape (224, 224, 1)
        test_image = x_test[0].astype(np.float32)
        # put the test image into a batch of size 1, shape (1, 224, 224, 1)
        test_image = np.expand_dims(test_image, axis=0)
    else:
        if test_onnx_batch_size == len(x_test):
            print(f"using full test set for onnx predictions")
            test_image = x_test.astype(np.float32)
        else:
            # get a batch of test images with shape (N, 224, 224, 1), evenly sampled
            x_test_size: int = len(x_test)
            batch_increment: int = int(x_test_size / test_onnx_batch_size)
            print(f"test_batch_size: {test_onnx_batch_size}, x_test_size: {x_test_size}, batch_increment: {batch_increment}")
            test_image = x_test[10:x_test_size:batch_increment].astype(np.float32)

    print(f"test_image shape: {test_image.shape}")

    # Get ONNX prediction
    ort_sess = ort.InferenceSession(onnx_model_filepath)
    onnx_pred_logits = ort_sess.run(None, {"input_tensor": test_image})[0]

    y_pred = np.argmax(onnx_pred_logits, axis=1)
    y_pred_onnx = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    # file_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    keras_model_datetime_parts: list = keras_model_filepath.split(".")[0].split("_") 
    file_datetime: str = "_".join(keras_model_datetime_parts[-2:])
    cm_filepath: str = f"results/figs/onnx/cm_onnx_subject_{data_subject_id}_{file_datetime}.png"

    confusion_matrix_title: str = f"ONNX Confusion Matrix"
    save_confusion_matrix_png(y_test, y_pred, cm_filepath, cm_title=confusion_matrix_title)
    print(f"Saved ONNX confusion matrix to: {cm_filepath}")
    
    # NOTE: the confusion matrices for ONNX and Keras are different, but they both have
    # 100% accuracy for class 0 (no others have 100% accuracy in both confusion matrices) 
    
if output_keras_confusion_matrix:
    # load example image from test data
    data_root_folder_path: str = config.default_dataset_path
    data_mode: str = "perp"
    data_subject_id: str = "4"
    data_subject: str = f"Subject_{data_subject_id}"
    data_image_size: int = 224

    (x_train, y_train), (x_test, y_test), data_num_classes = load_subject_arrays(
        data_root_folder_path, data_mode, data_subject, data_image_size
    )

    test_image = x_test.astype(np.float32)
    
    # keras_pred_logits = model.predict(test_image)
    # 4. Use direct call with training=False for the most accurate comparison
    # This bypasses the BatchNormalization training behavior
    keras_pred_logits = model(test_image, training=False).numpy()
    y_pred = np.argmax(keras_pred_logits, axis=1)
    y_pred_keras = y_pred

    # cm_filepath: str = f"results/figs/keras/cm_keras_subject_{data_subject_id}_{file_datetime}.png"

    cm = confusion_matrix(y_test, y_pred)

    cm_title = "Keras Confusion Matrix"

    print(f"{cm_title}:\n{cm}")
    
    # Count mismatched predictions
    num_mismatched_predictions: int = np.sum(y_pred_keras != y_pred_onnx)
    print(f"Number of mismatched predictions: {num_mismatched_predictions} out of {len(y_test)}")
    
