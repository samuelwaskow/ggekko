import sys
from flask import Flask, request, jsonify
from waitress import serve
import numpy as np

from agent import DuelingDDQNAgent
from env_daytrade import DaytradeEnvironment

app = Flask(__name__)
app.config["DEBUG"] = True
standard_features = ['MA4_MA8','MA4_MA12','STOCASTIC_MAIN','STOCASTIC_SIGNAL','RSI','CCI','MFI'],
timeframe = 'M1'
window = 2


#
# Initialize
#
def init():
    global winAgentBuy, wdoAgentBuy, winEnvBuy, wdoEnvBuy, winAgentSell, wdoAgentSell, winEnvSell, wdoEnvSell

    winEnvBuy = DaytradeEnvironment(None, 
                                window_size=window, 
                                features = standard_features,
                                buyer = True
                                )
    winAgentBuy = DuelingDDQNAgent(
                             input_dims=winEnvBuy.get_observation_space(),
                             n_actions=winEnvBuy.get_action_space(),
                             fname=f'../checkpoint/WIN_{timeframe}_BUY')
    winAgentBuy.load_model()
    wdoEnvBuy = DaytradeEnvironment(None, 
                                 window_size=window,
                                 features = standard_features,
                                 buyer = True
                                 )
    wdoAgentBuy = DuelingDDQNAgent(
                              input_dims=wdoEnvBuy.get_observation_space(),
                              n_actions=wdoEnvBuy.get_action_space(),
                              fname=f'../checkpoint/WDO_{timeframe}_BUY')
    wdoAgentBuy.load_model()

    winEnvSell = DaytradeEnvironment(None, 
                                window_size=window, 
                                features = standard_features,
                                buyer = False
                                )
    winAgentSell = DuelingDDQNAgent(
                             input_dims=winEnvSell.get_observation_space(),
                             n_actions=winEnvSell.get_action_space(),
                             fname=f'../checkpoint/WIN_{timeframe}_SELL')
    winAgentSell.load_model()
    wdoEnvSell = DaytradeEnvironment(None, 
                                 window_size=window,
                                 features = standard_features,
                                 buyer = False
                                 )
    wdoAgentSell = DuelingDDQNAgent(
                              input_dims=wdoEnvSell.get_observation_space(),
                              n_actions=wdoEnvSell.get_action_space(),
                              fname=f'../checkpoint/WDO_{timeframe}_SELL') 
    wdoAgentSell.load_model()


#
# Transform the input to the desired format
#
def transformInput(input):
    position = np.asfarray(input["position"])
    ma4ma8 = np.asfarray(input["ma4ma8"])
    ma4ma12 = np.asfarray(input["ma4ma12"])
    stocasticMain = np.asfarray(input["stocasticMain"])
    stocasticSignal = np.asfarray(input["stocasticSignal"])
    rsi = np.asfarray(input["rsi"])
    cci = np.asfarray(input["cci"])
    mfi = np.asfarray(input["mfi"])

    out = np.column_stack([
        ma4ma8, ma4ma12, stocasticMain, stocasticSignal, rsi, cci, mfi
    ])
    size = window * 7
    out = out.reshape(size)
    out = np.insert(out, 0, position[0])
    print(out)
    return out


#
# REST call
#
@app.route("/rest/<symbol>", methods=["POST"])
def predict(symbol):
    content = request.json
    input_content = transformInput(content)
    position = input_content[0]
    
    prediction = 1
    predictionBuy = 0
    predictionSell = 0

    if symbol == 'WIN':
        predictionBuy = int(winAgentBuy.advantage(input_content))
        predictionSell = int(winAgentSell.advantage(input_content))
    elif symbol == 'WDO':
        predictionBuy = int(wdoAgentBuy.advantage(input_content))
        predictionSell = int(wdoAgentSell.advantage(input_content))
    else:
        return 'Ativo Inválido', 400

    print(f'{symbol} - predictionBuy [{predictionBuy}] predictionSell [{predictionSell}]')
    if position > 0:
        prediction = 0 if predictionBuy == 0 else 1
    elif position < 0:
        prediction = 2 if predictionSell == 0 else 1
    else:
        if predictionBuy == 1 and predictionSell == 0:
            prediction = 2
        elif predictionSell == 1 and predictionBuy == 0:
            prediction = 0
        elif predictionSell == 1 and predictionBuy == 1:
            prediction = 3
        else:
            prediction = 1     
    
    print(f'{symbol} - position [{position}] prediction [{prediction}]')
    return jsonify({ "action": prediction })


#
# MAIN
#
if __name__ == "__main__":
    init()
    serve(app, host='127.0.0.1', port=5000)
    #app.run(threaded=True)
