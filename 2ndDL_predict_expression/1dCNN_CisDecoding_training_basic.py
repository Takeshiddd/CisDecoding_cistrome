import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import optimizers
from data_utils.generator import get_train_val_generator
from data_utils.generator import _split_train_val
from cnn_models.cnn_model_basic import build_cnn
from sklearn.utils.class_weight import compute_class_weight
import argparse
from tensorflow.python.keras import optimizers
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_channel', default=50, type=int, help='number of TF channels.')
    parser.add_argument('--data_length', default=20, type=int, help='length of sequences or bins.')
    parser.add_argument('--batch_size', default=156, type=int, help='batch size for training.')
    parser.add_argument('--epochs', default=10, type=int, help='number of epochs for training.')
    parser.add_argument('--val_rate', default=0.3, type=float, help='rate of validation data.')
    parser.add_argument('--shuffle', default=True, type=bool, help='phenotype data training shuffle')
    parser.add_argument('--class_weight', default=5, type=float, help='class-weight or positive sample imbalance rate')
    parser.add_argument('--target_file', default='BRup.txt', type=str, help='phenotype data file')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--out_file', default='model.h5', type=str, help='output model file name')
    parser.add_argument('--prediction_file', default='prediction.txt', type=str, help='output prediction file name')
    args = parser.parse_args()

    
    model = build_cnn(data_length=args.data_length, n_channel=args.n_channel)

    model.summary()
    class_weights = {0: 1, 1: args.class_weight}
    print("class weight")
    print(class_weights)
    model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Nadam(lr=args.learning_rate),
              metrics=['accuracy'])

    
    train_gen, valid_gen, val_targets = get_train_val_generator(args.batch_size, args.target_file, val_rate=args.val_rate, data_length=args.data_length, shuffle=args.shuffle)

    if valid_gen is None: val_step = None
    else: val_step = len(valid_gen)

    model.fit_generator(generator=train_gen
                        ,epochs=args.epochs
                        ,steps_per_epoch=len(train_gen)
                        ,verbose=1
                        ,validation_data=valid_gen
                        ,validation_steps=val_step
			,class_weight = class_weights)

    # Save model.
    model.save(args.out_file)
    prediction = model.predict_generator(valid_gen, val_step)
    val_pred = np.argmax(prediction, axis=1)
    np.set_printoptions(threshold=30000)
    pred=open(args.prediction_file, "w")
    pred.write(str(prediction))
    pred.close()
    
    # ROC-AUC
    print("Confusion Matrix")
    y_pred = np.argmax(prediction, axis=1)
    print(confusion_matrix(val_targets.argmax(axis=1), y_pred))

    print("ROC-AUC")
    y_pred_value = [prediction[i][1] for i in range(y_pred.shape[0])]
    roc = roc_auc_score(val_targets.argmax(axis=1), y_pred_value)
    print(roc)

    fpr, tpr, thresholds = roc_curve(val_targets.argmax(axis=1), y_pred_value)
    plt.plot(fpr, tpr, marker="o")
    plt.xlabel("FPR: False Positive Rate")
    plt.ylabel("TPR: True Positive Rate")
    plt.grid()
    plt.ylim([0.0,1.01])
    plt.figure(figsize= (3,2))
    plt.show()


