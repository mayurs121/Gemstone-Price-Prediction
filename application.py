from flask import Flask, request, render_template,jsonify
from flask_cors import CORS,cross_origin
from src.pipelines.prediction import Custom_Data, Prediction_Pipeline

application = Flask(__name__)

app = application

@app.route('/')
@cross_origin()
def home_page():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
@cross_origin()
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = Custom_Data(
            carat = float(request.form.get('carat')),
            depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z')),
            cut = request.form.get('cut'),
            color= request.form.get('color'),
            clarity = request.form.get('clarity')
        )

        prd_df = data.data_as_dataframe()
        
        print(prd_df)

        prd_pipeline = Prediction_Pipeline()
        pred = prd_pipeline.predict(prd_df)
        results = round(pred[0],2)
        return render_template('home.html',results=results,pred_df = prd_df)
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)