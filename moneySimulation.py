import tensorflow as tf
import pandas as pd
from collections import deque
import numpy as np
from sklearn import preprocessing
import dataCleaner as dC

# Good tickers so far; NGAS, SPX500, GBP/USD even though it was doing poorly over that time period

SEQ_LEN = dC.SEQ_LEN  # how long of a preceeding sequence to look at
FUTURE_PERIOD_PREDICT = dC.FUTURE_PERIOD_PREDICT  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = dC.RATIO_TO_PREDICT.split("old")[0]

MODEL_PATH = "models/EUR-USDold-TIK-20-SEQ-10-PRED-1593364221"

def preprocess_df(df):
    df = df.drop("bidclose", 1)
    df = df.drop("time", 1)
    df = df.drop("future", 1)

    for col in df.columns:  # go through all of the columns
        # normalize all
        df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
        df.dropna(inplace=True)  # remove the nas created by pct_change
        df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

    df.dropna(inplace=True)  # cleanup again... jic.

    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for j in df.values:  # iterate over the values
        prev_days.append([n for n in j])  # store all but the future
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append(np.array(prev_days))  # append those bad boys!

    X =[]

    # think should be able to skip this and just return sequential_data without a np.array() cast
    for seq in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences

    return X


def append_future_old(df):
    df['future'] = df['bidclose'].shift(-FUTURE_PERIOD_PREDICT)
    df['future'] = df['future'] - df[f'askclose']  # future is the price difference not the actual value, probably a cleaner way to do this
    # df['future'] = df['future'] - df['bidclose']  # This way doesnt include spread like above


def append_future(df):
    df.reset_index(inplace=True)  # give an index that increases by 1 and starts at 0
    index_of_bidclose = df.columns.get_loc("bidclose")
    future_vals = []
    for index, row in df.iterrows():
        future_vals.append(list(df.iloc[index:index + FUTURE_PERIOD_PREDICT, index_of_bidclose]))  # get the next few values (future to predict) maybe could find faster way

    df['future'] = future_vals


def prepare():
    dataset = f'crypto_data/{RATIO_TO_PREDICT}.csv'
    df = pd.read_csv(dataset, names=['time', 'askclose', 'high', 'open', 'bidclose', 'volume'])  # read in specific file sentdex uses low instead of askclose

    df.set_index("time", inplace=True)  # set time as index so we can join them on this shared time
    df = df[["askclose", "bidclose", "volume"]]  # ignore the other columns besides price and volume diff sentdex

    df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
    df.dropna(inplace=True)

    append_future(df)

    df.dropna(inplace=True)

    return df


dataframe = prepare()
current_close = dataframe["askclose"]
future_price_data = list(dataframe["future"])  # old version is just called future price since it would be a number rather than a list

data = preprocess_df(dataframe)

data = np.asarray(data)

#train_x, train_y, valx, valy = dC.sort_data()

# we need to load in the last 20 close prices in an np array
model = tf.keras.models.load_model(MODEL_PATH)

predictions = model.predict([data])

correct = 0
wrong = 0
total_money = 0
lowest_money = 0
highest_money = 0
number_of_positions = 0

for i in range(len(predictions)):
    prediction = predictions[i]
    future_prices = future_price_data[i]  # old version is just called future price since it would be a number rather than a list
    price_taken = 0
    # loop through the future values until we hit one that meets our profit, if none found we take price on final
    for j, val in enumerate(future_prices):
        if val > current_close[i] + 0.0003:
            # print(current_close[i], future_prices, val)
            price_taken = val  # we've run into a value which is take profit
            break
        if j == len(future_prices)-1 and price_taken == 0:  # If none of the next FUTURE values meet the profit we take the last
            price_taken = val
            # print(current_close[i], future_prices, val)

    future_price = price_taken - current_close[i]

    # print(prediction[0], prediction[1], future_price)
    if prediction[1] > prediction[0] and future_price < 0:  # prediction buy but went down
        wrong += 1
    elif prediction[1] > prediction[0] and future_price >= 0:  # prediction buy and went up
        correct += 1
    elif prediction[0] > prediction[1] and future_price < 0:  # prediction dont buy i think
        correct += 1
    elif prediction[0] > prediction[1] and future_price >= 0:  # prediction dont buy but went up
        wrong += 1

    if prediction[1] > 0.6:  # the confidence in prediction
        total_money += future_price * 100  # multiplication value is the lot size basically
        number_of_positions += 1

    if total_money < lowest_money:
        lowest_money = total_money

    if total_money > highest_money:
        highest_money = total_money

print(correct, wrong)
print(highest_money, lowest_money, number_of_positions)
print(f"£{total_money}")

