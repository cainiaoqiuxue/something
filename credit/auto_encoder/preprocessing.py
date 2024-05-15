import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras

from code.preprocessing import data_preprocess_v1


class AutoEncoder:
    def __init__(self):
        # self.ms = MinMaxScaler()
        self.ms = StandardScaler()
        self.train_feature = [f'n{i}' for i in range(1, 15)]
        self.df = None

    def get_data(self):
        feature, _ = data_preprocess_v1()
        drop_index = feature[self.train_feature].isna().any(axis=1)
        feature = feature.drop(feature.index[drop_index]).reset_index(drop=True)
        columns = feature.columns
        feature = self.ms.fit_transform(feature)
        feature = pd.DataFrame(feature, columns=columns)
        feature.fillna(-1, inplace=True)
        self.df = feature
        return feature

    def mask(self, df, p=0.5):
        mask_array = np.random.choice([np.NaN, 1], size=df.shape, p=[p, 1 - p])
        mask_df = df * mask_array
        mask_df = mask_df.fillna(-1)
        return mask_df

    def make_train_data(self):
        if self.df is not None:
            df = self.df
        else:
            df = self.get_data()
        label = df[self.train_feature]
        df[self.train_feature] = self.mask(df[self.train_feature])
        return df, label

    def ae_model(self, input_dim, hidden_dim, output_dim):
        '''
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(hidden_dim),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(output_dim)
        ])
        '''
        model=keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(256,activation='relu'),
            keras.layers.Dense(128,activation='relu'),
            keras.layers.Dense(output_dim)
        ])
        model.compile(loss=keras.losses.MeanSquaredError(),
                      optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),
                      metrics=['mse'])
        return model

    def train(self, epoch, save_model=False):
        early_stop = keras.callbacks.EarlyStopping(patience=10)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(patience=5)
        callbacks = [early_stop, reduce_lr]
        if save_model:
            save_path = '../save_model/tf_model2.hdf5'
            ckpt = keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, save_weights_only=True)
            callbacks.append(ckpt)

        for i in range(epoch):
            feature, label = self.make_train_data()
            train_x, test_x, train_y, test_y = train_test_split(feature, label, test_size=0.2, random_state=42)
            if i==0:
                model = self.ae_model(feature.shape[1], 20, label.shape[1])
            model.fit(train_x, train_y, epochs=1, callbacks=callbacks, validation_data=(test_x, test_y))


if __name__ == '__main__':
    import joblib

    ae = AutoEncoder()
    ae.train(50, save_model=False)
    # joblib.dump(ae.ms, '../save_model/ms.json')
