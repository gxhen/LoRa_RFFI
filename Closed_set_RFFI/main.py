import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score

from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop

from dataset_preparation import awgn, LoadDataset, ChannelIndSpectrogram

from deep_learning_models import classification_net
from keras.utils import to_categorical


def train(file_path_in,
          dev_range=range(0, 30),
          pkt_range=range(0, 1000)):

    """
    train_feature_extractor trains an RFF extractor using triplet loss.

    INPUT:
        FILE_PATH_IN is the path of training dataset.

        DEV_RANGE is the label range of LoRa devices to train the RFF extractor.

        PKT_RANGE is the range of packets from each LoRa device to train the RFF extractor.

        SNR_RANGE is the SNR range used in data augmentation.

    RETURN:
        MODEL is trained classification neural network.
    """

    # Load preamble IQ samples and labels.
    LoadDatasetObj = LoadDataset()
    data_train, label_train = LoadDatasetObj.load_iq_samples(file_path=file_path_in,
                                                             dev_range=dev_range,
                                                             pkt_range=pkt_range)

    # Shuffle the training data and labels.
    index = np.arange(len(label_train))
    np.random.shuffle(index)
    data_train = data_train[index, :]
    label_train = label_train[index]

    # One-hot encoding
    label_train = label_train - dev_range[0]
    label_one_hot = to_categorical(label_train)

    # Add noise to increase system robustness
    data_train = awgn(data_train, range(20, 80))

    # Convert to channel independent spectrogram
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_train)

    # Learning rate scheduler
    early_stop = EarlyStopping('val_loss', min_delta=0, patience=30)
    reduce_lr = ReduceLROnPlateau('val_loss', min_delta=0, factor=0.2, patience=10, verbose=1)
    callbacks = [early_stop, reduce_lr]

    # Specify optimizer and deep learning model
    opt = RMSprop(learning_rate=1e-3)
    model = classification_net(data.shape, len(np.unique(label_train)))
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    # Start training
    history = model.fit(data,
                        label_one_hot,
                        epochs=400,
                        shuffle=True,
                        validation_split=0.10,
                        verbose=1,
                        batch_size=32,
                        callbacks=callbacks)

    return model


def test(file_path_in,
         clf_path_in,
         dev_range=np.arange(0, 30),
         pkt_range=np.arange(0, 400)):

    """
    test_classification performs a classification task and returns the
    classification accuracy.

    INPUT:
        FILE_PATH_IN is the path of enrollment dataset.

        CLF_PATH_IN is the path of classification dataset.

        DEV_RANGE is the label range of LoRa devices.

        PKT_RANGE is the range of packets from each LoRa device.

    RETURN:
        ACC is the overall classification accuracy.
    """

    # Load preamble IQ samples and labels.
    LoadDatasetObj = LoadDataset()
    data_test, label_test = LoadDatasetObj.load_iq_samples(file_path=file_path_in,
                                                           dev_range=dev_range,
                                                           pkt_range=pkt_range)

    label_test = label_test - dev_range[0]

    # Load neural network
    net_test = load_model(clf_path_in, compile=False)

    # Convert to channel independent spectrogram
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_test)

    # Make prediction
    pred_prob = net_test.predict(data)
    pred_label = pred_prob.argmax(axis=-1)

    # Plot confusion matrix
    conf_mat = confusion_matrix(label_test, pred_label)
    classes = dev_range - dev_range[0] + 1

    plt.figure()
    sns.heatmap(conf_mat, annot=True,
                fmt='d', cmap='Blues',
                annot_kws={'size': 7},
                cbar=False,
                xticklabels=classes,
                yticklabels=classes)

    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('True label', fontsize=12)
    plt.savefig('confusion_matrix.pdf', bbox_inches='tight')
    plt.show()

    return accuracy_score(label_test, pred_label)


if __name__ == '__main__':

    # run_for = 'train'
    run_for = 'test'

    if run_for == 'train':

        file_path = './Train/dataset_training_aug.h5'

        clf_net = train(file_path)

        clf_net.save('cnn.h5')

    elif run_for == 'test':

        file_path = './Test/dataset_seen_devices.h5'
        clf_path = 'cnn.h5'

        acc = test(file_path, clf_path)
        print('Overall accuracy = %.4f' % acc)
