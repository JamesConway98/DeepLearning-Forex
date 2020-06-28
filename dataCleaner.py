from sklearn import preprocessing
import pandas as pd
from collections import deque
import random
import numpy as np

SEQ_LEN = 20  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 10  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "EUR-USDold"  # XAU-USD


def classify_old(current, future):
    if float(future) > float(current) + 0.000:  # if the future price is higher than the current+0.00045, that's a buy, or a 1
        return 1
    else:  # otherwise... it's a 0!
        return 0


# classifies as a 1 if somewhere in the next 10 <- (period to predict) it goes +0.003 profit
def classify(current, future_values):
    for i, value in enumerate(future_values):
        if float(value) > float(current) + 0.0003:  # if one of the future prices is higher than the current+0.0003, that's a buy, or a 1
            return 1

    # otherwise... it's a 0!
    return 0


def preprocess_df(df, validation=False):
    df = df.drop("future", 1)  # don't need this anymore.

    # I think we shoudld maybe drop bidclose and time here
    print ("DF:")
    print(df)

    for col in df.columns:  # go through all of the columns
        if col != "target":  # normalize all ... except for the target itself!
            df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
            df.dropna(inplace=True)  # remove the nas created by pct_change
            df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

    df.dropna(inplace=True)  # cleanup again... jic.

    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

    if not validation:
        #only randomise non validation data
        random.shuffle(sequential_data)  # shuffle for good measure.

        buys = []  # list that will store our buy sequences and targets
        sells = []  # list that will store our sell sequences and targets

        for seq, target in sequential_data:  # iterate over the sequential data
            if target == 0:  # if it's a "not buy"
                sells.append([seq, target])  # append to sells list
            elif target == 1:  # otherwise if the target is a 1...
                buys.append([seq, target])  # it's a buy!

        random.shuffle(buys)  # shuffle the buys
        random.shuffle(sells)  # shuffle the sells!

        lower = min(len(buys), len(sells))  # what's the shorter length?

        buys = buys[:lower]  # make sure both lists are only up to the shortest length.
        sells = sells[:lower]  # make sure both lists are only up to the shortest length.

        sequential_data = buys + sells  # add them together
        random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!


def append_target_single(df):
    df['future'] = df['bidclose'].shift(-FUTURE_PERIOD_PREDICT)  # add the value in future to close column
    df['target'] = list(map(classify_old, df['askclose'], df['future']))  # this takes into account the price we have to pay which is spread
    # df['target'] = list(map(classify, df['bidclose'], df['future'])) # can delte this when we are working out with spread
    df.dropna(inplace=True)


def append_target(df):
    df.reset_index(inplace=True)  # give an index that increases by 1 and starts at 0
    index_of_bidclose = df.columns.get_loc("bidclose")
    future_vals = []
    for index, row in df.iterrows():
        future_vals.append(list(df.iloc[index:index+FUTURE_PERIOD_PREDICT, index_of_bidclose]))  # get the next few values (future to predict) maybe could find faster way

    df['future'] = future_vals
    df['target'] = list(map(classify, df['askclose'], df['future']))  # this takes into account the price we have to pay which is spread
    df.dropna(inplace=True)


def sort_data():
    ratio = RATIO_TO_PREDICT.split('.csv')[0]  # split away the ticker from the file-name
    dataset = f'crypto_data/{ratio}.csv'  # get the full path to the file.
    df = pd.read_csv(dataset, names=['time', 'askclose', 'bidhigh', 'bidopen', 'bidclose', 'volume'])  # read in specific file, this is different from sentdex since he uses low instead of askclose

    df.set_index("time", inplace=True)  # set time as index so we can join them on this shared time
    df = df[["askclose", "bidclose", "volume"]]  # ignore the other columns besides price and volume

    df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
    df.dropna(inplace=True)

    # ask is what we buy it for, bid close is what we sell it for
    append_target(df)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.min_rows', 40)
    print(df)

    # here, split away some slice of the future data from the main df.
    times = sorted(df.index.values)
    # print(times, -int(0.05 * len(times)))
    last_5pct = sorted(df.index.values)[-int(0.05 * len(times))]

    validation_df = df[(df.index >= last_5pct)]
    df = df[(df.index < last_5pct)]

    print(df)

    train_x, train_y = preprocess_df(df)
    validation_x, validation_y = preprocess_df(validation_df, True)

    print(f"train data: {len(train_x)} validation: {len(validation_x)}")
    print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
    print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    validation_x = np.asarray(validation_x)
    validation_y = np.asarray(validation_y)

    return train_x, train_y, validation_x, validation_y

sort_data()