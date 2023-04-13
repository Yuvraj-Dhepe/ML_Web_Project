from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
from src.logger import logging

application = Flask(__name__)

app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                gender = request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch = request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=request.form.get('reading_score'),
                writing_score=request.form.get('writing_score'),
                
            )
            pred_df = data.get_data_as_data_frame()
            
            print(pred_df)
            
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template('home.html',results = results)
        
        except Exception as e:
            logging.error(f"Error occured while predicting the data:{e}")
            return render_template('home.html',results = e)

if __name__=='__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port) # This is the port assigned by the beanstalk, if any... else we use the 8080 port
    
    #app.run(host = '0.0.0.0',port = 8080) #this is the port number on which the application will run, so in docker run command we will map this port to the port of the container, ex. -p 8080:80 where 8080 is the port of the host and 80 is the port of the container


# Now the code has been finished, we have setup the ec2 instance, ecr repo, action-runners and setup the ec2 instance to recieve the action as soon as we commit the code and this will trigger our CI cd pipeline.
# No