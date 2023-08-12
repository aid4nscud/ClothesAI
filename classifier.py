from sklearn.preprocessing import OneHotEncoder
from keras_preprocessing.image import load_img, img_to_array
from keras.applications import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from kerastuner.tuners import RandomSearch
import joblib
from keras.applications.vgg16 import preprocess_input

# Read the CSV file
df = pd.read_csv("styles.csv", nrows=5000)
df['image'] = df.apply(lambda row: "images/" + str(row['id']) + ".jpg", axis=1)

# Convert the Target Columns to Strings
target_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage']
df[target_columns] = df[target_columns].astype(str)

# One-Hot Encode the Targets
encoder = OneHotEncoder()
targets = encoder.fit_transform(df[target_columns]).toarray()

# Split into Training and Validation Sets
train_df, val_df, train_targets, val_targets = train_test_split(df, targets, test_size=0.2, random_state=42)

# Function to Create Batches
def create_batches(data_df, targets, batch_size=32, target_size=(96, 96)):
    while True:
        for i in range(0, len(data_df), batch_size):
            batch_df = data_df.iloc[i:i + batch_size]
            images = [img_to_array(load_img(img_path, target_size=target_size)) for img_path in batch_df['image']]
            images = np.array(images)
            images = preprocess_input(images)  # Preprocess the images
            batch_targets = targets[i:i + batch_size]
            yield images, np.array(batch_targets)

# Define the model-building function for Keras Tuner
def build_model(hp):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
    for layer in base_model.layers:
        layer.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(hp.Int('units', min_value=256, max_value=1024, step=128), activation='relu')(x)
    output = Dense(targets.shape[1], activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[0.001, 0.0001, 0.00001])),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

batch_size = 32
train_gen = create_batches(train_df, train_targets, batch_size=batch_size)
val_gen = create_batches(val_df, val_targets, batch_size=batch_size)
# Perform hyperparameter tuning using Keras Tuner
tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=5, executions_per_trial=1,
                     directory='hyperparameter_tuning', project_name='my_tuning')
tuner.search(train_gen, epochs=10, validation_data=val_gen, validation_steps=len(val_df) // batch_size)

# Get the best model from the tuner
best_model = tuner.get_best_models(num_models=1)[0]

# Save the encoder and best model
encoder_path = 'encoder.pkl'
joblib.dump(encoder, encoder_path)

best_model.save('best_model.keras')

print("Model trained successfully!")
