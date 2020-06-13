from flask import render_template,Flask,request,url_for
from flask_bootstrap import Bootstrap

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

app=Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	df=pd.read_csv("data/dataset1.csv")
	df_X=df.name
	df_Y=df.sex
	corpus=df_X
	cv=CountVectorizer()
	X=cv.fit_transform(corpus)
	nb_model=open("models/naivebayesgendermodel.pkl","rb")
	clf=joblib.load(nb_model)

	if request.method=='POST':
		name_q=request.form['namequery']
		data=[name_q]
		vect=cv.transform(data).toarray()
		pred=clf.predict(vect)

	return render_template('results.html',prediction=pred,name=name_q.upper())



	

	
if __name__ =='__main__':
	app.run(debug=True)
