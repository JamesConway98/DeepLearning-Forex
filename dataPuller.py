import fxcmpy
import socketio
import pandas
import apiKey
from datetime import datetime

TRADING_API_URL = "https://api-demo.fxcm.com"
WEBSOCKET_PORT = 443
TOKEN = apiKey.get_fxcm_api_key()
RATIO = 'NGAS' #'GBP/USD'
START_DATE = '27/05/18 10:55:00'

con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error', server='demo', log_file='log.txt')
print(con.get_instruments_for_candles())


def get_data(start, stop, append=True):
    data = con.get_candles(RATIO, period='m1', start=start, stop=stop)

    # data.rename(columns={"bidclose": f"{RATIO}_close", "tickqty": f"{RATIO}_volume"}, inplace=True)
    # data.drop(columns=['askopen', 'askclose', 'askhigh', 'asklow'], inplace=True)
    print(list(data.columns.values))

    data = data[['askclose', 'bidhigh', 'bidopen', 'bidclose', 'tickqty']]  # this is different from sentdex version since he has bidlow first instead of askclose
    print(data)
    data.index = data.index.map(lambda date: int(datetime.timestamp(date)))

    print(data)

    mode = 'w'
    if append:
        mode = 'a'
    data.to_csv(rf'pulled_data\{RATIO.replace("/", "-")}.csv', mode=mode, header=False)


start_date_str = START_DATE
start_date_time_obj = datetime.strptime(start_date_str, '%d/%m/%y %H:%M:%S')
start_timestamp = int(start_date_time_obj.timestamp())

# there are 60 seconds between each point and we can max pull 10000
stop_timestamp = start_timestamp + (60 * 10000)
stop_date_time_obj = datetime.fromtimestamp(stop_timestamp)

print(f'Starting timestamp: {start_timestamp}, Stopping timestamp: {stop_timestamp}')

get_data(start_date_time_obj, stop_date_time_obj, False)

for i in range(30):

    start_timestamp = start_timestamp + (60 * 10000)
    start_date_time_obj = datetime.fromtimestamp(start_timestamp)

    stop_timestamp = stop_timestamp + (60 * 10000)
    stop_date_time_obj = datetime.fromtimestamp(stop_timestamp)

    get_data(start_date_time_obj, stop_date_time_obj)

con.close()