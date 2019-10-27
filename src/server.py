from flask import Flask
import requests
import json 
import HurriModel
import county_heat

app = Flask(__name__)

@app.route('/', methods=['GET'])
def helloworld():
  return 'Hello, World!'


@app.route('/ai', methods=['POST'])
def ai():
  try:
    HurriModel.some_test()
    
    paths = []
    

    paths.append({'coordinates': {'lat': 22.5, 'lng': -93.5}, 'windspeed': 30, 'distance': 0, 'direction': 0})
    paths.append({'coordinates': {'lat': 22.7, 'lng': -93.8}, 'windspeed': 30, 'distance': 23.604, 'direction': 140.568})
    paths.append({'coordinates': {'lat': 23.1, 'lng': -94.6}, 'windspeed': 35, 'distance': 57.935, 'direction': 161.351})
    paths.append({'coordinates': {'lat': 23.5, 'lng': -95.4}, 'windspeed': 40, 'distance': 57.802, 'direction': 174.302})
    paths.append({'coordinates': {'lat': 23.9, 'lng': -96.3}, 'windspeed': 45, 'distance': 63.292, 'direction': 337.251})
    paths.append({'coordinates': {'lat': 24.1, 'lng': -96.3}, 'windspeed': 45, 'distance': 46.294, 'direction': 340.709})
    paths.append({'coordinates': {'lat': 24.9, 'lng': -98.6}, 'windspeed': 30, 'distance': 62.892, 'direction': 237.544})
    paths.append({'coordinates': {'lat': 25.2, 'lng': -99.8}, 'windspeed': 25, 'distance': 77.920, 'direction': 258.374})
    print(paths)
    return json.dumps(paths)
  except Exception as e:
    print(e)
    return 'What happened?'

@app.route('/extra', methods=['POST'])
def extra():
  try:

    heat = county_heat.model() # Doesn't give me any data fudge

    return "So it works I guess " + json.dumps(heat)
  except:
    return "It didn't work"








if __name__ == '__main__':
  app.run(debug=False, port= 3001)

