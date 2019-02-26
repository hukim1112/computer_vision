import os
import tensorflow as tf
import numpy as np
import pandas as pd

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path


def load_data(y_name='Species'):
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop('Species')

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop('Species')

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    # This is input function for training
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    return dataset.shuffle(1000).repeat().batch(batch_size)


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


(train_x, train_y), (test_x, test_y) = load_data()
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns,
                                        hidden_units=[10, 10],
                                        n_classes=3)


# Train the Model.
classifier.train(
    input_fn=lambda: train_input_fn(train_x, train_y, batch_size=32),
    steps=100)

# Evaluation
eval_result = classifier.evaluate(
    input_fn=lambda: eval_input_fn(test_x, test_y, batch_size=32))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


# Prediction
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}

predictions = classifier.predict(
    input_fn=lambda: eval_input_fn(predict_x,
                                   labels=None,
                                   batch_size=32))

template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print(template.format(
        SPECIES[class_id], 100 * probability, expec))
