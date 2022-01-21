import json
import pickle
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from main import treshold


# 2. Create the app object
app = FastAPI()
#pickle_in = open(r"C:\Users\Utilisateur\OneDrive\Bureau\PROJET7\classifier.pkl","rb")
with open(r"C:\Users\Utilisateur\OneDrive\Bureau\PROJET7\classifier.pkl", 'rb') as f:
    classifier = pickle.load(f)
#classifier=pickle.load(pickle_in)
#


# pickle_in = open("classifier.pkl","rb")
# classifier=pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, WorldAA'}

# class Client(BaseModel):
#     name: Optional[str]=""
#     id: float


# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/credit/{client_id}')
def get_client_data(client_id:int):
    #client_id=client.id
    data=pd.read_csv(r"C:\Users\Utilisateur\OneDrive\Bureau\PROJET7\valid.csv", encoding='cp1252')
    if data.shape[-1]==770:
        data.set_index('SK_ID_CURR', inplace=True)
    index=np.array(data.loc[client_id]).reshape(1, -1)
    x_valid=np.load(r'C:/Users/Utilisateur/OneDrive/Bureau/PROJET7/val.npy')
    prediction=classifier.predict(index)
    y_valid = np.load(r'C:/Users/Utilisateur/OneDrive/Bureau/PROJET7/yvalid.npy')
    res = treshold(classifier, x_valid, data, y_valid)
    print(res)
    result=classifier.predict_proba(index)
    if result[0][1]>res:
        dict_final = {
            'prediction': int(prediction),
            'proba_remboureser': float(result[0][1]),
            'treshold':float(res),
            'message':'On est d\ésol\é  monsieur, on ne peut pas accepter votre  demande de  cr\édit:' f'{result}'}
        return json.dumps(dict_final)

    else:
        dict_final = {
            'prediction': int(prediction),
            'proba_remboureser': float(result[0][1]),
            'treshold': float(res),
            'message': 'F\élicitations  monsieur, votre  demande de  crédit est  accept\ée:' f'{result}'}
        return json.dumps(dict_final)


# @app.put("/credit/{client_id}")
# def update_item(client_id: int, client: Client):
#     return {"item_name": client.name, "item_id": client_id}



# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=80)
