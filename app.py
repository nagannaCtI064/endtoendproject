import json
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))
scale=pickle.load(open("scale.pkl","rb"))
encoder=pickle.load(open("encoder.pkl","rb"))
@app.route("/")
def home():
    return render_template("home.html")
import json
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))
scale=pickle.load(open("scale.pkl","rb"))
encoder=pickle.load(open("encoder.pkl","rb"))
@app.route("/")
def home():
    return render_template("home.html")
@app.route("/predict",methods=["POST"])
def predict():
    if request.method=="POST":
        age = request.form.get("Age")
        Gender = request.form.get("Gender")
        Platform = request.form.get("Platform")
        Daily_Usage_Time_minutes = request.form.get("Daily_Usage_Time (minutes)")
        Posts_Per_Day = request.form.get("Posts_Per_Day")
        Likes_Received_Per_Day = request.form.get("Likes_Received_Per_Day")
        Comments_Received_Per_Day = request.form.get("Comments_Received_Per_Day")
        Messages_Sent_Per_Day = request.form.get("Messages_Sent_Per_Day")
        
        data=[age,Gender,Platform,Daily_Usage_Time_minutes,Posts_Per_Day,Likes_Received_Per_Day,Comments_Received_Per_Day,Messages_Sent_Per_Day]
        columns=['Age', 'Gender', 'Platform', 'Daily_Usage_Time (minutes)',
        'Posts_Per_Day', 'Likes_Received_Per_Day', 'Comments_Received_Per_Day',
        'Messages_Sent_Per_Day']
        df=pd.DataFrame(data=[data],columns=columns)
        df["Gender"]=encoder.fit_transform(df["Gender"])
        df["Platform"]=encoder.fit_transform(df["Platform"])
        final_input=scale.fit_transform(df)
        output=model.predict(final_input)
        output_list=output.tolist()
        return jsonify(output_list)



if __name__=="__main__":
    app.run(debug=True)    
@app.route("/predict_api",methods=["POST"])
def predict_api():
    data=[float(x) for x in request.form.values()]
    final_input=scale.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    return jsonify(output)

if __name__=="__main__":
    app.run(debug=True)    