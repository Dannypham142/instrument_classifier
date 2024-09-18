import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, LayerNormalization, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from scipy.io import wavfile
from kapre import STFT, Magnitude, MagnitudeToDecibel
from kapre.composed import get_melspectrogram_layer
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from glob import glob
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

sample_rate = 16000
delta_time = 1.0
batch_size = 16
src_root = 'wavfiles_cleaned'

def train():
    log_path = os.getcwd() + '/conv1d_history.csv'
    params = {'N_CLASSES':len(os.listdir(src_root)),
              'SAMPLE_RATE':sample_rate,
              'DELTA_TIME':delta_time}
    model_type = 'conv1d'
    models = {'conv1d':Conv1D(**params)} # Add Conv2D and lSTM in the future??
    csv_path = os.path.join('logs', '{}_history.csv'.format(model_type))
    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [wav.replace(os.sep, '/') for wav in wav_paths if '.wav' in wav]
    classes = sorted(os.listdir(src_root))
    le = LabelEncoder()
    le.fit(classes)
    labels = [os.path.split(wav)[0].split('/')[-1] for wav in wav_paths]    
    labels = le.transform(labels)
    wav_train, wav_val, label_train, label_val = train_test_split(wav_paths,
                                                                  labels,
                                                                  test_size=0.1,
                                                                  random_state=42)
    train_group = DataGenerator(wav_train, label_train, sample_rate, delta_time,
                       params['N_CLASSES'], batch_size=batch_size)
    val_group = DataGenerator(wav_val, label_val, sample_rate, delta_time,
                       params['N_CLASSES'], batch_size=batch_size)
    model = models[model_type]
    checkpoint = ModelCheckpoint('models/{}.h5'.format(model_type), monitor='val_loss',
                         save_best_only=True, save_weights_only=False,
                         mode='auto', save_freq='epoch', verbose=1)
    csv_logger = CSVLogger(csv_path, append=False)
    model.fit(train_group, validation_data=val_group,
              epochs=30, verbose=1,
              callbacks=[csv_logger, checkpoint])

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sample_rate, delta_time, n_classes, batch_size=32, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.delta_time = delta_time
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    
    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batchsize))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1*self.batch_size)]
        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # Initializing empty arrays for time data
        X = np.empty((self.batch_size, int(self.sr*self.dt), 1), dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)
            X[i,] = wav.reshape(-1,1)
            Y[i,] = to_categorical(label, num_classes=self.n_classes)
        return X, Y

    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

def Conv1D(N_CLASSES=10, SAMPLE_RATE=16000, DELTA_TIME=1.0):
    input_shape = (int(SAMPLE_RATE * DELTA_TIME), 1)

    model = Sequential()
    model.add(get_melspectrogram_layer(input_shape=input_shape,
                                       n_fft=512,
                                       win_length=400,
                                       hop_length=160,
                                       pad_end=True,
                                       sample_rate=SAMPLE_RATE,
                                       n_mels=128,
                                       return_decibel=True,
                                       input_data_format='channels_last',
                                       output_data_format='channels_last'))
    model.add(LayerNormalization(axis=2, name='batch_norm'))
    model.add(TimeDistributed(layers.Conv1D(8, kernal_size=(4), activation='tanh'), name='td_conv_1d_tanh'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_1'))
    model.add(TimeDistributed(layers.Conv1D(16, kernal_size=(4), activation='relu'), name='td_conv_1d_relu_1'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_2'))
    model.add(TimeDistributed(layers.Conv1D(32, kernal_size=(4), activation='relu'), name='td_conv_1d_relu_2'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_3'))
    model.add(TimeDistributed(layers.Conv1D(64, kernal_size=(4), activation='relu'), name='td_conv_1d_relu_3'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_4'))
    model.add(TimeDistributed(layers.Conv1D(128, kernal_size=(4), activation='relu'), name='td_conv_1d_relu_4'))
    model.add(layers.GlobalMaxPooling2D(name='global_max_pooling_2d'))
    model.add(layers.Dropout(rate=0.1, name='dropout'))
    model.add(layers.Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense'))
    model.add(layers.Dense(N_CLASSES, activation='softmax', name='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    train()