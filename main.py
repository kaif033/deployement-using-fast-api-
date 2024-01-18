import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
import pickle

from BankNotes import BankNote

app=FastAPI()
pickle_in = open(r'C:\Users\HP\OneDrive\Pictures\Saved Pictures\fast api\rf.pkl', 'rb')

classifier=pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message':'helooo bhai seekh rha hu '}

@app.get('/Welcome')
def  get_name(name : str):
    return {'wlecome to my learning tutorial':f'{name}'}

@app.post('/predict')
def predict_species(data:BankNote):
    data=data.dict()

    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
    pred=classifier.predict([[variance,skewness,curtosis,entropy]])
    if pred[0]>0.5:
        pred="fake note"
    else:
        pred="fare one"
    return pred


if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)
