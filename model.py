#importing necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.models import Sequential, load_model
from keras.layers import Dropout, Dense, GlobalMaxPooling2D,GlobalAveragePooling2D,Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout, Dense,LSTM



#Model Architecture
def CNN_RNN():


	model = Sequential()
	model.add(TimeDistributed(Conv2D(64,(3,3),activation='relu'),input_shape=(1,150,150,3)))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))

	model.add(TimeDistributed(Conv2D(32,(3,3),activation='relu')))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))


	model.add(TimeDistributed(Flatten()))

	#RNN
	model.add(LSTM(100,return_sequences=False))


	model.add(Dense(4,activation='softmax'))


	#Getting model summary
	print(model.summary())

	return model




def DCNN_():
    input_shape=(150,150,3)

    base_cnn = VGG16( weights='imagenet', include_top=False, input_shape=input_shape)

    model = Sequential()
    model.add(base_cnn)
    # don't train existing weights
    for layer in base_cnn.layers:
        layer.trainable = False

    model.add(GlobalMaxPooling2D(name="gap"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(4, activation="softmax"))

    return model