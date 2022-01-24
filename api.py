import json
import pickle
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from model import treshold

# 2. Create the app object
app = FastAPI()

with open(r"C:\Users\Utilisateur\OneDrive\Bureau\PROJET7\classifier.pkl", 'rb') as f:
    classifier = pickle.load(f)

data = pd.read_csv(r"C:\Users\Utilisateur\OneDrive\Bureau\PROJET7\valid.csv", encoding='cp1252')
x_valid = np.load(r'C:/Users/Utilisateur/OneDrive/Bureau/PROJET7/val.npy')
y_valid = np.load(r'C:/Users/Utilisateur/OneDrive/Bureau/PROJET7/yvalid.npy')
res = treshold(classifier, x_valid, data, y_valid)




# 3. Index route, opens automatically on http://127.0.0.1:80
@app.get('/')
def index():
    return {'message': 'Hello, WorldAA'}

@app.get('/credit/{client_id}')
def get_client_data(client_id: int):
    # client_id=client.id
    if data.shape[-1] == 770:
        data.set_index('SK_ID_CURR', inplace=True)

    index = np.array(data.loc[client_id]).reshape(1, -1)
    prediction = classifier.predict(index)
    print(res)
    result = classifier.predict_proba(index)
    if result[0][1] > res:
        dict_final = {
            'prediction': int(prediction),
            'proba_remboureser': float(result[0][1]),
            'treshold': float(res),
            'message': 'On est désolé  monsieur, on ne peut pas accepter votre  demande de  crédit:' f'{result}'}
        return json.dumps(dict_final)

    else:
        dict_final = {
            'prediction': int(prediction),
            'proba_remboureser': float(result[0][1]),
            'treshold': float(res),
            'message': 'Félicitations  monsieur, votre  demande de  crédit est  acceptée:' f'{result}'}
        return json.dumps(dict_final)



# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:80
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=80)
