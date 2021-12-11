import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt


def MakeModel(layerlist, xshape, yshape, activation='relu',droprate=0.5):
    num_layerlist = len(layerlist)
    model = keras.Sequential()
    model.add(keras.Input(shape=xshape))
    for i,nodes in enumerate(layerlist):
        model.add(Dense(i,activation=activation))
        model.add(Dropout(droprate)) # dropout on only the hidden layers
    model.add(Dense(yshape,activation=None))
    return model

if __name__ == "__main__":

    numcats = 4
    bigDataScaled = pd.read_pickle("bigData_scaled.pkl")
    bigDataScaled.drop(columns= bigDataScaled.select_dtypes(exclude=[np.number,bool]).columns, inplace=True) # drop nonnumerical columns other than test designator
    data = bigDataScaled

    # prepare datas
    X_trains = [pd.DataFrame] * numcats
    Y_trains = [pd.DataFrame] * numcats
    for category in range(numcats):
        X_train = data[np.logical_and(data['category']==category, [not x for x in data['Test']==False])].copy(deep=True)
        X_train.drop(columns=['category', 'Test'], inplace=True)
        Y_train = data['Pr'][np.logical_and(data['category']==category, [not x for x in data['Test']==False])].copy(deep=True)
        X_trains[category] = X_train.copy(deep=True)
        Y_trains[category] = Y_train.copy(deep=True)

    def makeNTrain(category, layerlist, activation='relu',droprate=0.5, epochs=10):
        X_train = X_trains[category]
        Y_train = Y_trains[category]
        xshape = X_train.shape[1]
        yshape = Y_train.shape[0]
        model = MakeModel(layerlist, xshape, yshape, activation, droprate)
        model.compile(optimizer='Adam', loss='mse', metrics=['mse', 'mae'])
        callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min", restore_best_weights=True)
        training = model.fit(x=X_train, y=Y_train, epochs = epochs, validation_split=0.15, callbacks = [callback])
        return model, training


    droprates = [0.25, 0.35, 0.5, 0.65, 0.75]
    archs = [[12, 12, 10, 10, 9],[15, 15, 15, 15],[12, 10, 8, 6, 4],[10, 15, 15, 10, 5]]

    # find best droprate w predetermined arch
    dropscores = np.zeros((numcats, len(droprates)))
    mindrops = np.zeros(numcats)
    for category in range(numcats):
        for i,droprate in enumerate(droprates):
            model, training = makeNTrain(category, archs[0], droprate=droprate)
            dropscores[category, i] = training.history["val_loss"][-1]
        mindrops[category] = droprates[np.argmin(dropscores[category, :])]

    # use that droprate and find best arch
    archscores = np.zeros((numcats, len(archs)))
    minarchs = np.zeros(numcats)
    for category in range(numcats):
        for i,arch in enumerate(archs):
            model, training = makeNTrain(category, archs[0], droprate=mindrops[category])
            archscores[category, i] = training.history["val_loss"][-1]
        minarchs[category] = archs[np.argmin(archscores[category, :])]

    # finally train all the models
    models = []
    trainings = []
    Pr_mses = []
    Pr_maes = []
    for category in range(numcats):
        model,training = makeNTrain(category, minarchs[category], droprate=droprates[category], epochs=30)
        model.save('model'+str(category))
        models.append(model)
        trainings.append(training)

        X_test = data[data['category']==category and data['test']==True]
        X_test.drop(columns=['category', 'test'], inplace=True)
        Y_test = data['Pr'][data['category']==category and data['test']==True]
        Y_hat = model.predict(X_test)
        Pr_mses.append(keras.losses.MSE(Y_test, Y_hat))
        Pr_maes.append(keras.losses.MAE(Y_test, Y_hat))

        f=plt.figure(category)
        plt.title("category"+str(category))
        plt.plot(training.history["loss"], label = "loss")
        plt.plot(training.history["val_loss"], label = "val_loss")
        plt.xlabel("epoch")
        plt.ylabel("mse")
        plt.show()
        f.savefig(f"plots/traininghistory{category}.png")

np.save("model_mses.arr", Pr_mses)
np.save("model_maes.arr", Pr_maes)