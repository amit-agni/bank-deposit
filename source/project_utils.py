import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.impute
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import precision_score, recall_score,confusion_matrix,f1_score


def dp_dropCols(data,cols):
    data.drop(columns = cols,inplace = True)

def dp_ImputeNumericCols(data,strategy="median"):
    numericCols = data.select_dtypes(include=["int64","float64"]).columns   
    save_dtypes = dict(data.select_dtypes(include=["int64","float64"]).dtypes)

    imputer = sklearn.impute.SimpleImputer(strategy=strategy) # create a SimpleImputer instance
    imputer.fit(data[numericCols]) #Creates imputer.statistics_ that stores the "strategy=median" values
    
    #transform creates np array, convert to dataframe
    data[numericCols] = pd.DataFrame(imputer.transform(data[numericCols]), columns=numericCols,index=data.index)
    data[numericCols] = data[numericCols].astype(save_dtypes)

    return(data)

def dp_EncodeOrdinalCols(data,cols,categories= 'auto'):
    ordEnc = sklearn.preprocessing.OrdinalEncoder(categories=categories)    
    if(len(cols) ==1):        
        # ordEnc.fit(data[cols].values.reshape(-1,1))
        # outArray = ordEnc.transform(data[cols].values.reshape(-1,1))        
        outArray = ordEnc.fit_transform(data[cols].values.reshape(-1,1)) #Alternate way
    else:
        # ordEnc.fit(data[cols])
        # outArray = ordEnc.transform(data[cols])        
        outArray = ordEnc.fit_transform(data[cols]) #Alternate way                

    data[cols] = pd.DataFrame(outArray, columns=cols,index=data.index).astype('int')

    return(data)

def dp_EncodeOneHot(data,cols,drop=None,handle_unknown="error"):
    """
    Implemented as a Class, this function can be ignored
    """

    for i in cols:        
        oheEnc = sklearn.preprocessing.OneHotEncoder(drop=drop,dtype="int",handle_unknown=handle_unknown,sparse=False)
        outArray = oheEnc.fit_transform(data[i].values.reshape(-1,1))
        colNames = i + data[i].unique()
        print(colNames)
        data.drop(columns = i,inplace = True)        
        # print(pd.DataFrame(outArray,columns=i + df[cols].unique()))
        data = pd.concat([data,pd.DataFrame(outArray,columns=colNames)],axis=1) #ignore_index=True)

    return(data)


class DataframeOneHotEncoder(BaseEstimator,TransformerMixin):
    """
    TO DO :
        1. Only drop is handled. Remaining arguments not handed
        2. Separate fit and transform. Such that the fit can be applied to train and then use it to transform test
        [Like this](https://stackoverflow.com/questions/44601533/how-to-use-onehotencoder-for-multiple-columns-and-automatically-drop-first-dummy/44601764)
        3. Implement inverse transform

    Example :
        cols = data.select_dtypes(include="object").columns
        enc = DataframeOneHotEncoder(cols)
        data = enc.transform(data)    

    """
    def __init__(self,cols,drop=None,sparse=False,handle_unknown="error",dtype="int"):                
        self.cols = cols
        self.drop = drop
        self.sparse = sparse
        self.handle_unknown = handle_unknown
        self.dtype = dtype

    def fit(self):
        return self

    def transform(self,data):
        for i in self.cols:        
            oheEnc = sklearn.preprocessing.OneHotEncoder(drop=self.drop,sparse=self.sparse,handle_unknown=self.handle_unknown,dtype=self.dtype)
            oheEnc.fit(data[i].values.reshape(-1,1))
            outArray = oheEnc.transform(data[i].values.reshape(-1,1))            
            if self.drop != None:
                # colNames = i + data[i].unique() 
                colNames = np.array(*[i + s for s in oheEnc.categories_]) #Better to use the instance variable
                colNames = colNames[1:]
            else:
                # colNames = i + data[i].unique()
                colNames = np.array(*[i + s for s in oheEnc.categories_])

            data.drop(columns = i,inplace = True)        
            data = pd.concat([data,pd.DataFrame(outArray,columns=colNames)],axis=1) 
        return(data)



def dp_StandardiseNumericCols(data,cols):
    std = sklearn.preprocessing.StandardScaler()
    std.fit(data[cols])
    data[cols] = pd.DataFrame(std.transform(data[cols]))
    return(data)


def countsAndProportions(arr):    
    s1 = arr.value_counts()
    s2 = arr.value_counts()/len(arr)       
    print(pd.concat([s1,s2],axis=1))


def valueCountsPerColumn(data):
    """    
    a = valueCountsPerColumn(train_set_X[cols_OHE]).set_index(["feature","value"])
    b = valueCountsPerColumn(test_set_X[cols_OHE]).set_index(["feature","value"])
    c = pd.concat([a,b],ignore_index=False,axis=1)

    c[c.isna().any(axis=1)]
    """

    cols = data.columns
    out = pd.DataFrame()
    for i in cols:    
        temp = pd.DataFrame(data[i].value_counts()).rename_axis("value").reset_index() #Convert row names to column
        temp.rename(columns = {temp.columns[1]:"count"},inplace= True) #Rename columns by position
        temp.insert(0,"feature", i) #Insert column by position
        out = out.append(temp)

    out.reset_index(drop = True,inplace = True)  #inplace needed as reset_index does not modify the DataFrame      
    return out


#################### Modeling #############################


def classificationMetrics(y_true,y_prob):
    print("Confusion Matrix :\n", confusion_matrix(y_true,y_prob))
    print("Precision :",precision_score(y_true,y_prob))
    print("Recall :",recall_score(y_true,y_prob))
    print("F1 Score :",f1_score(y_true,y_prob))
    print("******************************************")
    print("y_true Counts :")
    print(countsAndProportions(y_true))
    print("pred Counts :")
    print(countsAndProportions(pd.Series(y_prob)))


def modelTrainPredict(model,train_prep,train_set_Y,test_set_X,test_set_Y):
    model = model
    model.fit(train_prep, train_set_Y)
    test_prep = full_pipeline.transform(test_set_X)
    pred = model.predict(test_prep)
    classificationMetrics(test_set_Y,pred)
    