from flask import Flask, jsonify, request
from classifier import  get_prediction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



app= Flask (__name__)



#Fetching the data
X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)



#Splitting the data and scaling it
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=3500, test_size=500)



@app.route("/predict-alphabet",methods=["POST"])
def predict_data():
    # image = cv2.imdecode(np.fromstring(request.files.get("digit").read(), np.uint8), cv2.IMREAD_UNCHANGED)
    image=request.files.get("alphabet")
    prediction=get_prediction(image)

    return jsonify({
        "prediction":prediction
    }),200


if __name__=="__main__":
    app.run()
