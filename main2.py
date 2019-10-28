import os
import tensorflow as tf
import numpy as np
import Preprocess2
import Model

batch_size_n = 10
file_name_csv = 'first_6000_training_labels.csv'
IMAGE_FOLDER = '../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/*.dcm'

# all_labels = Preprocess2.load_csv(file_name_csv)
train  = Preprocess2.read_dcm_files(IMAGE_FOLDER)
print(f'original shape: {Preprocess2.get_datas(train)[0].pixel_array.shape}')


# print(f'all labels: {len(all_labels)}')
# print(f'all training images: {len(train)}')
# # --------------------------
# checkpoint_path = "training/cp_{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# # Create a callback that saves the model's weights every 5 epochs
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path, 
#     verbose=1, 
#     save_weights_only=True,
#     period=1)

# # --------------------------
# training_images, training_labels = Preprocess2.load_data(batch_index=0, BATCH_SIZE=batch_size_n, all_labels=all_labels, train=train, quite=False)
# print(f'training_images[0].shape: {training_images[0].shape}')
# model = Model.get_model1(training_images)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# for batch_index in range(1, 11):
#     model.fit(x=training_images, y=training_labels, epochs=2, validation_split=0.2, shuffle=True,
#             callbacks=[cp_callback], verbose=0)
#     model.save_weights(checkpoint_path.format(epoch=0))
#     training_images, training_labels = Preprocess2.load_data(batch_index=batch_index, BATCH_SIZE=batch_size_n, all_labels=all_labels, train=train, quite=False)


# Save the weights using the `checkpoint_path` format
# model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the new callback
# model.fit(train_images, 
#               train_labels,
#               epochs=50, 
#               callbacks=[cp_callback],
#               validation_data=(test_images,test_labels),
#               verbose=0)
# -------------------------------
# Groupby filters to get the correct category number.
# filter_df = labels.groupby('PatientID').sum()
# filter_brain = filter_df[filter_df['Label'] > 0]
# filter_has_brain = labels[labels['PatientID'].isin(filter_brain['PatientID'])].drop(columns='ID').drop(labels.columns[0], axis=1)
