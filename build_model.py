from keras.callbacks import *

from keras.layers import *
from keras.models import Model, Sequential
from keras.optimizers import Adam, Adadelta, SGD

from gradient_reversal import GradientReversal

from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from math import exp


class BuildModel:
    def __init__(self, config_path, checkpoint_path, n_timesteps, tokenizer,
                 vocab_dim=64, n_symbols=100, input_length=100, embedding_weights=None):

        self.config = config_path
        self.checkpoint_path = checkpoint_path
        self.tokenizer = tokenizer
        self.n_timesteps = n_timesteps
        self.vocab_dim = vocab_dim
        self.n_symbols = n_symbols
        self.embedding_weights = embedding_weights
        self.input_length = input_length

    def create_loss_weights(self):
        """Create loss weights that increase exponentially with time.

        Returns
        -------
        type : list
            A list containing a weight for each timestep.
        """
        weights = []
        for t in range(self.n_timesteps):
            weights.append(exp(-(self.n_timesteps - t)))
        return weights

    def build_bert_model(self):
        bert_model = load_trained_model_from_checkpoint(self.config, self.checkpoint_path)
        for l in bert_model.layers:
            l.trainable = True
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        #         x3_in = Input(shape=(5,))

        x = bert_model([x1_in, x2_in])
        x = Lambda(lambda x: x[:, 0])(x)

        #         x = concatenate([x, x3_in], axis=-1)

        p = Dense(2, activation='softmax')(x)

        self.model = Model([x1_in, x2_in], p)
        self.sample_compile_model()

    def build_text_cnn_model(self):
        self.model = Sequential()  # or Graph or whatever
        # model.add(Input(shape=(input_length,)))
        self.model.add(Embedding(output_dim=self.vocab_dim,
                            input_dim=self.n_symbols,
                            mask_zero=True,
                            weights=[self.embedding_weights],
                            input_length=self.input_length, trainable=False))  # Adding Input Length
        self.model.add(Convolution1D(256, self.vocab_dim, border_mode='same', subsample_length=1, trainable=True))
        self.model.add(Activation('relu'))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))
        self.sample_compile_model()


    def sample_compile_model(self):
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(1e-5),
            metrics=['accuracy']
        )
        self.model.summary()

        return self.model

    def compile_model(self, loss='categorical_crossentropy',
            optimizer=SGD(lr=0.001), metrics=None, loss_weights=None):

        if metrics is None:
            metrics = ['accuracy']
        if loss_weights is None:
            weights = self.create_loss_weights()
            loss_weights = {'domain_classifier': weights, 'aux_classifier': weights}
        loss_demo = {'domain_classifier': loss, 'aux_classifier': loss}
        self.model.compile(loss=loss_demo, optimizer=optimizer, metrics=metrics, loss_weights=loss_weights)
        print(self.model.summary())
        return self.model

    def build_bert_domain_model(self):
        bert_model = load_trained_model_from_checkpoint(self.config, self.checkpoint_path)
        for l in bert_model.layers:
            l.trainable = True
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))

        x = bert_model([x1_in, x2_in])
        x = Lambda(lambda x: x[:, 0])(x)

        flip_layer = GradientReversal(0.31)
        p_in = flip_layer(x)

        des1 = Dense(256, activation='relu')(p_in)
        des1 = Dropout(0.2)(des1)
        des1 = Dense(64, activation='relu')(des1)
        des1 = Dropout(0.2)(des1)

        des2 = Dense(256, activation='relu')(x)
        des2 = Dropout(0.2)(des2)
        des2 = Dense(64, activation='relu')(des2)
        des2 = Dropout(0.2)(des2)
        des2 = Dense(16, activation='relu')(des2)
        des2 = Dropout(0.2)(des2)

        p2 = Dense(5, activation='softmax', name='domain_classifier')(des1)

        p = Dense(2, activation='softmax', name='aux_classifier')(des2)

        self.model = Model([x1_in, x2_in], [p, p2])
        self.compile_model()

    def model_fit_1(self, train_D, valid_D):
        checkpointer = ModelCheckpoint(filepath="./checkpoint_bert.hdf5",
                                       monitor='val_acc', verbose=True, save_best_only=True, mode='auto')

        early = EarlyStopping(monitor='val_acc', patience=4, verbose=0, mode='auto')
        reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)

        self.model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=100,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[reducelr, checkpointer, early],
            verbose=True
        )

    def model_fit_2(self, train_D, valid_D):
        checkpointer = ModelCheckpoint(filepath="./checkpoint_bert.hdf5",
                                       monitor='val_aux_classifier_acc',
                                       verbose=True, save_best_only=True,
                                       mode='auto')

        #         early = EarlyStopping(monitor='val_aux_classifier_loss', patience=4, verbose=0,
        #                               mode='auto')
        #         reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
        # model = self.build_bert_domain_model()

        self.model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=200,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[checkpointer],
            verbose=True
            )

    # def cnn_model_fit(self, train_D, val_D):


