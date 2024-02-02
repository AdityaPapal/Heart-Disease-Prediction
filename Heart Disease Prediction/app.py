import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('trained_model.pkl', 'rb'))


def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,-1)
    loaded_model = pickle.load(open("trained_model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
    

        if int(result) == 0:
            return render_template("Result.html", final_result="Congratulations Patient wealth is Good! ...")
        elif result == 1:
            return render_template("Result.html", final_result="Patient has a heart Disease!....")


if __name__== "__main__":
    app.run(debug=True)