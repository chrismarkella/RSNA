def load_data():
    (training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    print(f'# of training images: {len(training_images)}')
    print(f'# of test images: {len(test_images)}')
    print(f'training images shape: {training_images.shape}')
    print(f'test images shape: {test_images.shape}')
    return (training_images, training_labels), (test_images, test_labels)


def reshaping(training_images, test_images):
    training_images = training_images.reshape(*training_images.shape, 1)
    test_images = test_images.reshape(*test_images.shape, 1)
    print(f'training images shape: {training_images.shape}')
    print(f'test images shape: {test_images.shape}')
    return training_images, test_images


def get_model1():
    model = tf.keras.models.Sequential()

    layers = [
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=training_images.shape[1:]),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')]

    for layer in layers:
        model.add(layer)

    model.summary()

    return model


def get_model2():
    model = tf.keras.models.Sequential()

    layers = [
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=training_images.shape[1:]),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')]

    for layer in layers:
        model.add(layer)
    return model


def get_model3():
    model = tf.keras.models.Sequential()

    layers = [
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=training_images.shape[1:]),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')]


def get_model4():
    model = tf.keras.models.Sequential()

    layers = [
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=training_images.shape[1:]),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')]

    for layer in layers:
        model.add(layer)
    return model


def get_models():
    models = [
            get_model1(),
            get_model2(),
            get_model3(),
            get_model4(),
    ]
    return models


def train_models(models, epochs):
    for model in models:
        model.summary()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x=training_images, y=training_labels, batch_size=100, epochs=epochs, validation_split=0.2, shuffle=True)
        test_loss, test_accuracy = model.evaluate(x=test_images, y=test_labels)
        print(f'test loss: {test_loss}, test accuracy: {test_accuracy}')
