from flask import Flask, request, render_template, url_for, send_from_directory
import numpy as np
import pandas as pd

app = Flask(__name__)

# def SelectColumns(DF):
    # columnList = ['FILENUM', 'PROD_DATE', 'P(TurnOff)','AREA', 'PROSPECT', 'PAD', 'WELL_NUMBER','PLAY', 'GAS_MCF', 'WATER_BBL','CHOKE_SETTING',
       # 'TUBING_PRESSURE_AVG', 'CASING_PRESSURE_AVG',
       # 'TEMPERATURE_FLOWLINE', 'DIFFERENTIAL_PRESSURE', 'STATIC_PRESSURE']
    # return DF[columnList]

# @app.route("/test")
# def test():
    # df = pd.read_csv("project/inference.csv")
    # df = SelectColumns(df)
    # return df.to_html(classes='Seneca')

# @app.route("/")
# def showTable():
    # df = pd.read_csv("project/inference.csv")
    # df = SelectColumns(df)
    # return render_template('view.html', tables=[df.to_html(classes='Seneca',float_format=lambda x: '%10.2f' % x)],
     # titles = ['Current Inference from Neural Network'])

# @app.route("/prospect", defaults={'probability': 0.0})
# @app.route("/prospect/<float:probability>")
# def ShowTables(probability=0.0):
    # DF = pd.read_csv("project/inference.csv")
    # DF = SelectColumns(DF)
    # prospectNames = [prospect for prospect in DF['PROSPECT'].unique() if prospect!='-1']
    # tables=[]
    # titles=['Inference']
    # for prospect in prospectNames:
        # tempDF = DF[DF['PROSPECT']==prospect]
        # tempDF = tempDF[tempDF['P(TurnOff)'] > probability]
        # tempDF = tempDF.sort_values('P(TurnOff)', ascending=False)
        # if tempDF.shape[0] > 0:
            # tables.append(tempDF.to_html(classes='Seneca', float_format=lambda x: '%10.2f' % x))
            # titles.append(prospect)

    # return render_template('view.html', tables=tables, titles = titles)

# @app.route("/download.csv")
# def downloadFile():
    # return send_from_directory(directory='../', filename='inference.csv')

@app.route("/maxrebo", methods=['GET', 'POST'])
def home():
    return render_template("home.html")
   
@app.route("/greeting")
def greeting():
    return "<h1 style='color:green'>Hello World!</h1>"

if __name__ == "__main__":
    app.run(host='0.0.0.0')