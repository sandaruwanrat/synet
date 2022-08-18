import numpy as np
import pandas as pd
import tensorflow as tf


gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)


from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import re
#from keras import models
#from keras import layers
#from keras import optimizers
#import tensorflow as tf
#from keras.models import load_model

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow as tf
from tensorflow.keras.models import load_model


#from keras.callbacks import ModelCheckpoint
#from keras.callbacks import EarlyStopping
from sklearn import preprocessing
#from keras.models import load_model
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import make_classification
#from sklearn.metrics import ConfusionMatrixDisplay
from Bio import SeqIO
#from fasta_one_hot_encoder import FastaOneHotEncoder

print("start prediction")


########################################################################
#load model
modelp=load_model('/scrfs/storage/ratnayak/data/synet/Arb/MYB/MYB_mod/MYB_32bp_v2_19TF_cmbneg_rnd6k.h5')


#read one sequcen at a time and gives predictions
#find a less time consuming algorithm.


#fasta = []# keep this soutside to loop allover the genes and finally get prediction
testpred = []


fasta = []# keep this soutside to loop allover the genes and finally get prediction

gname=[]
#test = []



#########

tragetgenes=[]

targetwindow=[]

finalpred=[]


for seq_record in SeqIO.parse("DEG_DE.2kb_complete.promoters_Nfilt.fa", "fasta"):
    
    print("Promoter SEQ \n",seq_record.seq)
    k = 0
    
    #fasta.append(seq_record.seq)
    gname.append(seq_record.id)
    
    print(">" + str(seq_record.id) + "\n")
    
    windowpred=[]
    
    #loop over one promoter at a time
    while (k < (len(seq_record.seq)-32)) :
        print("index",k)
        #print(">" + str(seq_record.id) + "\n")
        #print(str(seq_record.seq[i:i+201]) + "\n")
        #wind_seq=str(seq_record.seq[i:i+201] + "\n")
        wind_seq=str(seq_record.seq[k:k+32])
        print(len(seq_record.seq[k:k+32]))
        
        onehtlist=[]

        
        
        sequence = wind_seq
        
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
        
        
        if encodeln == 4:
            print(onehot_encoded_seq[1])
            #onehtout=onehot_encoded_seq[1]
            #print(onehot_encoded_seq)
            onehtlist.append(onehot_encoded_seq)
            #allylist.append(y)
        else:
            pass
        
        
        
        
        
        nl=len(onehtlist)
        for i in range(nl):
            kk=onehtlist[i]
            print(np.shape(kk))
        
        #isinstance(onehtlist, list)
        all_oneht_arr=np.array(onehtlist)
        #print("gene targets",all_oneht_arr)
        print("shape of the matrix",np.shape(all_oneht_arr))
        batchsize=np.shape(all_oneht_arr)
        
        
        ###############################################
        threshold = 0.8
        
        if batchsize[0] > 0:
            result = modelp.predict(all_oneht_arr, verbose=2)
            result = result > threshold
            
        else:
            pass
        #print(result)


        result_map = list(map(int, result ))

        #store predcitons for all the windows    
        # testpred.append(result_map)
        #store the predcitons for a one sequce windows
        windowpred.append(result_map)
        
        print(result_map)
        #print("final pred",result_map[0])
        ################################################
        #this is not suitable for the test but will work for the final predcition script
        #if result_map[0]==1 :
         #   print(seq_record.id)
         #   tragetgenes.append(seq_record.id)
         #   targetwindow.append(k)
        
        #else :
        #    pass
        #print(testpred)


         
        #print("window pred",windowpred)
        #print("inside while loop")
        #fasta.append(wind_seq)
        k += 32
        

    #print("out of while loop")
    #print("window pred out of while llop",windowpred)
    #unravel, or flatten
    windowpredarr=np.array(windowpred)
    wp=windowpredarr.flatten()
    dd=1 in wp
    
    if dd == True:
        pr=1
        finalpred.append(pr)
    else :
        pr=0
        finalpred.append(pr)



print(finalpred)
print("Done for loop")
#print(finalpred)
    
#tgenesarray=np.array(tragetgenes)
#indexarray=np.array(targetwindow)
    
#tgdf = pd.DataFrame({'tgenes':tgenesarray, 'indexarray':indexarray})
    
#tgdf.to_csv('SOY_Pred_targets100W.csv')
    #geneseqdf = pd.DataFrame(data=geneseqarray, columns=["seq"])
    #print(geneseqdf)
    
#SEQarr=np.array(fasta)
GENEarr=np.array(gname)

genecount=len(GENEarr)

synetprearr=np.array(finalpred)

#ylb=np.ones(genecount)


#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.datasets import make_classification
#from sklearn.metrics import ConfusionMatrixDisplay
#
#print(classification_report(ylb,synetprearr))
#
#cm = confusion_matrix(y_true=ylb, y_pred=synetprearr)
#
#print(cm)



tgdf = pd.DataFrame({'Targets':GENEarr,'Model_pred':synetprearr})
    
tgdf.to_csv('ARB_MYB_32bp_cross_Pred_targets32W_DEGgenes_thresh0.8_0609.csv')
    
 










