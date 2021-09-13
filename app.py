from flask import Flask, render_template, request, jsonify
import pandas as pd
import xgboost as xgb
app = Flask(__name__, template_folder='template')
#model = pickle.load(open('xgb_best14.pkl', 'rb'))
#with open('new_xgb_model','rb') as read_file:
#    model = pickle.load(read_file)
model = xgb.XGBRegressor()
model.load_model("xgb_best14.json")

@app.route("/")

@app.route("/home")
def home():
    return render_template("home.html")
 
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/result")
def cancer():
    return render_template("result.html")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

@app.route("/graph")
def graph():
    return render_template("output.html")



@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        series = {#'Deaerator6Level ': [float(request.form["Deaerator6Level"])],
                'BfwFlowToSuperheater': [float(request.form["BfwFlowToSuperheater"])],
                'SteamDesuperheaterCont': [float(request.form["SteamDesuperheaterCont"])],
                'BoilerFeedWater': [float(request.form["BoilerFeedWater"])],
                'MainGasInletToFurnace': [float(request.form["MainGasInletToFurnace"])],
                'FuelGasBehindCv': [float(request.form["FuelGasBehindCv"])],
                'CombustionAirFlow': [float(request.form["CombustionAirFlow"])],
                'AirBurnerBox': [float(request.form["AirBurnerBox"])],
                'MainSteamTemperature': [float(request.form["MainSteamTemperature"])],
                #'FlueGasFurnace': [float(request.form['FlueGasFurnace'])],
                'Boiler6FlueGasOutlet': [float(request.form["Boiler6FlueGasOutlet"])],
                'SteamBoiler': [float(request.form["SteamBoiler"])],
                #'WindBoxPressure': [float(request.form["WindBoxPressure"])],
                'CombustionAir ': [float(request.form["CombustionAir"])],
                #'Boiler6SteamDrum': [float(request.form["Boiler6SteamDrum"])],
                'MainSteamHeader': [float(request.form["MainSteamHeader"])],
                'EconomizerWaterInlet ': [float(request.form["EconomizerWaterInlet"])],
                'EconomizerWaterOutlet ': [float(request.form["EconomizerWaterOutlet"])]
           }
        
        vector = pd.DataFrame(series)
     
        prediction = model.predict(vector)
        output = round(prediction[0],2)
        if output<2:
            return render_template('prediction.html',prediction_text="O2 Content dibawah 2%")
        elif output>10:
            return render_template('prediction.html',prediction_text="O2 Content diatas 10%")
        else:
            return render_template('prediction.html',prediction_text="Nilai kandungan O2 Content adalah {:.2f}".format(output))
    else:
        return render_template('prediction.html')



if __name__=="__main__":
    app.run(debug=True)




