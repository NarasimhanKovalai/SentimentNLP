import uvicorn
from class_for_testing import AirlineFeedback
from fastapi import FastAPI
import pickle
from main import prediction

app=FastAPI()
pickle_in=open("sentiment_analysis_model.pkl","rb")
classifier=pickle.load(pickle_in)


#base endpoint which is the simplest API Endpoint
@app.get('/')

def index():
    return{'message': 'Hello, there'}

#endpoint to greet users
@app.get('/Welcome')
def get_name(name:str):
    return{'Welcome to my website':f'{name}'}

#endpoint which takes a text as input and outputs sentiment associated with it (positive or negative)
@app.post('/predict')
def predict_sentiment(data :AirlineFeedback):
    print(data,type(data))
    data=data.dict()
    print(data,type(data))
    text=data['text']
    print(text,type(text))
    prediction=classifier._test_single_feedback(text)
    print(prediction)
    if (prediction[0]==0):
        ans='negative'
    else:
        ans='positive'
    
    #return dictionary in JSON format - key value pair
    return {
      
        'Sentiment':ans 
    }
    
#run the main instance in local host
if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1',port=8010)





