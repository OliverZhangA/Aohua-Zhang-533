import pandas as pd
import json
import requests
from pandas.tseries.offsets import BDay

a1 = -0.01
n1 = 3

def test_hw2():
    df = pd.read_csv('ex_data.csv')
    df1 = df
    print(df)
    df['Date'] = pd.DatetimeIndex(df['Date']) + BDay()
    print(df)
    # df.index = df.index.map(lambda x: x + 0 * BDay())
    # submitted entry orders
    submitted_entry_orders = []
    for index, row in df.iterrows():
        price = (1 + a1) * float(row['Close'])
        submitted_entry_orders.append([index, row['Date'], 'IVV', 'BUY', 'LMT', str(price), 'SUBMITTED'])

    for order in submitted_entry_orders:
        print(order)
    return submitted_entry_orders, df1

def calculate_cancelled(submitted_entry_orders, history):
    size = len(submitted_entry_orders)
    for i in range(size):
        # if the lowest price in future n days is smaller than submitted price, fill
        # else cancel
        submitted_entry_orders[i][6] = 'CANCELLED'
        for j in range(1, n1+1):
            if i + 1 + j > size:
                submitted_entry_orders[i][6] = 'SUBMITTED'
                break
            # df.iloc[[4]]
            if float(history.iloc[[i + j]]['Low']) > float(submitted_entry_orders[i][5]):
                submitted_entry_orders[i][6] = 'EXIT'
    for order in submitted_entry_orders:
        print(order)

    return submitted_entry_orders


if __name__ == '__main__':
    submitted_orders, history = test_hw2()
    print('_____________________')
    calculate_cancelled(submitted_orders, history)
    # test_hw2()
    # getnext_workday()
