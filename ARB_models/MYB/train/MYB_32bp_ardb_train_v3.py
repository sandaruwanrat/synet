###MYB

import numpy as np
import pandas as pd
import tensorflow as tf


gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)


import tensorflow as tf
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import classification_report
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from tensorflow.keras.layers import TimeDistributed
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, confusion_matrix, log_loss
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
#from fasta_one_hot_encoder import FastaOneHotEncoder

print("start")







##########Read the fasta file,there are better fucntions from fasxid and biopython

def readfas(fasname):
    fasta = []
    test = []
    with open(fasname) as file_one:
        for line in file_one:
            line = line.strip()
            if not line:
               continue
            if line.startswith(">"):
                active_sequence_name = line[1:]
                if active_sequence_name not in fasta:
                    test.append(''.join(fasta))
                    fasta = []
                continue
            sequence = line
            fasta.append(sequence)

    # Flush the last fasta block to the test list
    if fasta:
        test.append(''.join(fasta))


    #np.shape(test)
    ln=len(test)
    newtest=test[1:ln]
    seqarray=np.array(newtest)
    return seqarray
#print(seqarray)
####################################################################################




##############################this need to to be automated to use the file name and do everyth
#in a fucntions

######################positive data


posseqarray=readfas("pos_MYB_ARB_FAM19TF_32bp_nFILT.fa")
lenpos=len(posseqarray)



dfposseq = pd.DataFrame(data=posseqarray, columns=["seq"])

ypos=pd.DataFrame(np.ones(lenpos))

dfposseq["ylabel"]=ypos
print(dfposseq)

#############negative
negseqarray=readfas("neg_MYB_rnd6kcmb_FAM19TF_32bp_nFILT.fa")
lenneg=len(negseqarray)

dfnegseq = pd.DataFrame(data=negseqarray, columns=["seq"])

yneg=pd.DataFrame(np.zeros(lenneg))

dfnegseq["ylabel"]=yneg
print(dfnegseq)


########################
df_all = pd.concat([dfposseq, dfnegseq])
np.shape(df_all)
print(df_all)


####shuffle
df_all_shuf=df_all.iloc[np.random.permutation(len(df_all))]
df_all_shuf['ylabel'] = df_all_shuf['ylabel'].astype('category')
print(df_all_shuf)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
################onehotencode
onehtlist=[]
allylist=[]

sqln=len(df_all_shuf)
#sqln=len(dfnegseq)

print(sqln)

for s in range(sqln):
#for s in range(0,10):
    #print(s)
    
    sdf_ins=df_all_shuf.iloc[s]

    sequence = sdf_ins[0]
    y=sdf_ins[1]

    #get sequence into an array
    seq_array = array(list(sequence))
    #print(seq_array)
    #integer encode the sequence
    label_encoder = LabelEncoder()
    integer_encoded_seq = label_encoder.fit_transform(seq_array)
    #print(integer_encoded_seq)
    #one hot the sequence
    onehot_encoder = OneHotEncoder(sparse=False)
    #reshape because that's what OneHotEncoder likes
    integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
    #print(integer_encoded_seq)
    onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
    #print(onehot_encoded_seq[1])
    
    encodeln=len(onehot_encoded_seq[1])
#     onehtlist.append(onehot_encoded_seq)
#     allylist.append(y)
    if encodeln == 4:
        #print(onehot_encoded_seq[1])
        #print(onehot_encoded_seq)
        onehtlist.append(onehot_encoded_seq)
        allylist.append(y)
    else:
        pass
        
    #print(np.shape(onehot_encoded_seq))
    
    
#updated=positive_oneht
#finallist.append(temp)
print("done")


all_oneht_arr=np.array(onehtlist)
print("Xdata",all_oneht_arr)
np.shape(all_oneht_arr)

###################################################


#####################################labling

allylist_arr=np.array(allylist)
print("ydata",allylist_arr)
ly=len(allylist_arr)

print("ydata shape",np.shape(allylist_arr))
#allylist_arr=allylist_arr.reshape((ly, 1))
#print(allylist_arr)

ydata=np.reshape(allylist_arr, (ly, 1))



lb = preprocessing.LabelBinarizer()

ylb=lb.fit_transform(ydata)

print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<y Lable>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>...",ylb)



######################################################







###split train 70% and 30%(val+test)


X_train, X_testval, y_train, y_testval = train_test_split(all_oneht_arr, ylb, test_size = 0.3)


print("X train shape",np.shape(X_train))

print("X testval shape",np.shape(X_testval))

print("y train shape",np.shape(y_train))

print("y testval shape",np.shape(y_testval))



lranget=len(y_train)

lrangett=len(y_testval)

print("progress>>>>>>>>>>>>")


#########################################################################################

#####split test set and validation set 15% each

# Use train_test_split() Method.
X_val, X_test, y_val, y_test = train_test_split(X_testval,y_testval, test_size=0.5)
#print(train)

print("X val shape",np.shape(X_val))

print("X test shape",np.shape(X_test))

print("y val shape",np.shape(y_val))

print("y test shape",np.shape(y_test))





######################################<<<<<<<<< model>>>>>>>>>>>>>>

model = models.Sequential()


#number of epochs to train for
nb_epoch = 100
#amount of data each iteration in an epoch sees
#54
batch_size = 1000

#filter size 32 gives the best
#opt = tf.keras.optimizers.SGD(learning_rate=0.01)

opt1 = tf.keras.optimizers.Adam(learning_rate=0.001)

model.add(layers.Conv1D(filters=32, kernel_size=10, activation='relu', padding='same',input_shape=(32,4)))
model.add(layers.Dropout(rate=0.1))

#model.add(layers.Dense(units=1,activation='relu'))
model.add(layers.Conv1D(filters=16,kernel_size=8, padding='same',activation='relu'))
model.add(layers.Dropout(rate=0.2))
model.add(TimeDistributed(Dense(64,activation='relu')))
#model.add(layers.Dropout(rate=0.2))
model.add(Bidirectional(LSTM(64,return_sequences=True)))
model.add(Dropout(0.5))
model.add(layers.Flatten())
#give more than two class prediction
#model.add(layers.Dense(1, activation='relu'))
#model.add(layers.Dense(1, activation='softmax'))
model.add(layers.Dense(1, activation='sigmoid'))




#model.compile(optimizer=opt ,loss='CategoricalCrossentropy',
    #metrics=['accuracy'])
#50%
#model.compile(optimizer='adam' ,loss='CategoricalCrossentropy',
#metrics=['accuracy'])
#94%accuacry
model.compile(optimizer=opt1 ,loss='BinaryCrossentropy',
metrics=['accuracy'])


#model.compile(optimizer=opt1,loss='CategoricalCrossentropy',
#metrics=['accuracy'])


#def custom_loss(y_true, y_pred):
 #   return tf.compat.v1.losses.sigmoid_cross_entropy(y_true, y_pred, label_smoothing=0.1)


#model.compile(optimizer='adam',loss=custom_loss,metrics=['accuracy'])



print(model.summary())


####early stopping
#patience=20
patience=20

checkpointer = ModelCheckpoint(filepath='/scrfs/storage/ratnayak/data/synet/Arb/MYB/MYB_mod/MYB_32bp_v3_19TF_cmbneg_rnd6k.h5',verbose=1, save_best_only=True)

earlystopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)





history=model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,verbose=1, validation_data=( X_val, y_val),callbacks=[checkpointer, earlystopper])

score = model.evaluate(X_val, y_val, verbose=0)

print('Test score:', score[0])
print('validation Test accuracy:', score[1])


#_, accuracy = model.evaluate(X_test, X_test)
print('validation Test Accuracy: %.2f' % (score[1]*100))



######################################
#output_dir="dnnmodels"

#model_json = model.to_json()
#output_json_file = open(output_dir + '/model24.json', 'w')
#output_json_file.write(model_json)
#output_json_file.close()

#model.save('dnnmodels/fixed_AT5G24110_modelv6_bl.h5')

#import matplotlib.pyplot as plt
#fig = plt.figure()
#pyplot.plot(history.history['loss'])
#pyplot.plot(history.history['accuracy'])
#pyplot.title('model loss vs accuracy')
#pyplot.xlabel('epoch')
#pyplot.legend(['loss', 'accuracy'], loc='upper right')
#
#fig.savefig("model_los_acc_v1.png")



#print ('Saving final model')
model.save('/scrfs/storage/ratnayak/data/synet/Arb/MYB/MYB_mod/MYB_32bp_v3_19TF_cmbneg_rnd6k.h5', overwrite=True)






##########################################
print(" Saving done.....")



pred_result = model.predict(X_test)

print("pred>>>",pred_result)

print("pred shape",np.shape(pred_result))

rpred=np.round(np.ravel(pred_result),decimals=0)
print("rpred>>>",rpred)


class_preds = np.argmax(pred_result, axis=-1)
print("class_preds ",class_preds )

print("class_preds shape",np.shape(class_preds))



print("ytest", y_test)
rytest=np.ravel(y_test)
#print("ydata shape",np.shape(y_data))
print("<<<<<<<<<<<<<<<<< classification report using rounded pred >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(classification_report(rytest,rpred))
#

print("<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")




print("<<<<<<<<<<<<<<<<< classification threhhold >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#print(classification_report(rytest,class_preds))
#

threshold = 0.8

result = model.predict(X_test, verbose=2)
result = result > threshold

print(result)


result_map = list(map(int, result ))

print(result_map)
print("<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

print("<<<<<<<<<<<<<<<<< classification report using bool vals >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(classification_report(rytest,result_map))



#################################################
print(confusion_matrix(rytest,result_map))

cm = confusion_matrix(y_true=rytest, y_pred=result_map)

ax = sns.heatmap(cm/np.sum(cm), annot=True,
            fmt='.2%', cmap='Blues')

ax.set_title(' Confusion Matrix\n\n');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])


fig = ax.get_figure()
#figure.savefig('svm_conf.png', dpi=400)

#fig=plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
fig.savefig("MYB_ARBD_wrky_32bp_19TF_cmbneg_rnd6k_v3.png")

def visualize_loss_acc(history):
#import matplotlib.pyplot as plt

# Setting Parameters
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))
    fig1 = plt.figure()
    # 1) Accracy Plt
    plt.plot(epochs, acc, 'g' ,label = 'training acc')
    #plt.plot(epochs, loss, 'r' ,label = 'training loss')
    plt.plot(epochs, val_acc, 'r' , label= 'validation acc')
    plt.title('Train-Validation_acc')
    plt.ylim(0,1.1)
    plt.legend(loc='best')
    #plt.show()
    fig1.savefig("myb_acc_curve_v3.png")
    
    
    
    fig2 = plt.figure()
    plt.plot(epochs, loss, 'g' ,label = 'training loss')
    #plt.plot(epochs, val_acc, 'bo' , label= 'validation acc')
    plt.plot(epochs, val_loss, 'r' , label= 'validation loss')
    plt.title('Train-Validation_loss')
    #plt.legend(bbox_to_anchor=(0, 0), loc='best')
    plt.legend(loc='best')
    fig2.savefig("myb_loss_curve_v3.png")
    #plt.show()
    
visualize_loss_acc(history)
#########################################################################
#precision, recall, _ = precision_recall_curve(rytest, result)
#
#fig2=plt.figure()
#plt.step(recall, precision, color='b', alpha=0.2, where='post')
#plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.ylim([0.0, 1.05])
#plt.xlim([0.0, 1.05])
#
#plt.title("auPRC: " + str(average_precision_score(rytest, result)))
#fig2.savefig("auPCR_arb_mod_train.png")
