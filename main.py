import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

response = " "
production1 = " "
app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def nameroute():
    global response, production1
    if (request.method == 'GET'):
        request_data = request.data
    # request_data = json.loads(request_data.decode('utf-8'))
    # location = request_data['city']
        user_api = "765fb6f4e326058496208ad4b995e5a0"
        location = "kottayam"
        complete_api_link ="https://api.openweathermap.org/data/2.5/weather?q=" + location + "&appid=" + user_api
        api_link = requests.get(complete_api_link)
        api_data = api_link.json()
        # create variables to store and display data
        print("!@#", api_data)
        temp = ((api_data['main']['temp']) - 273.15)
        print(temp, "is ")
        humid = api_data['main']['humidity']
        wind = api_data['wind']['speed']
        rain = api_data['clouds']['all']
        df = pd.read_csv("cropsetss")
        df.head()
        features = df[['Rainfall', 'Temperature', 'Humidity', 'Windspeed']]
        target = df['Crop']
        labels = df['Crop']
        X_train, X_test, y_train, y_test = train_test_split(features, target,test_size=0.2, random_state=2)
        RF = RandomForestClassifier(n_estimators=100, random_state=1)
        RF.fit(X_train, y_train)
        y_pred = RF.predict(X_test)
        predicted_values = RF.predict(X_test)
        x = metrics.accuracy_score(y_test, predicted_values)
        df = np.array([[temp, humid, rain, wind]])
        prediction = RF.predict(df)
        # print(prediction)
        new = pd.read_csv("cropandprodctns", index_col='Crop')
        pg = new.loc[prediction]
        n = pg.Production
        production = int(n)
        area = 2563
        if area > 0:
            crop_yield = production / area

            print("::::::",prediction, crop_yield)
            return jsonify({'response': response, 'production': production1, })
        # print(crop_yield)
        else:
          return "Some value is missing"
    else:
      return jsonify({'response': response, 'production': production1, })


if __name__ == "__main__":
    app.run(debug=True)
