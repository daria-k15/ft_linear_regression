import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from util import *

class linear_regression:
    def __init__(self, po, pn, hs, live, l_rate, iter):
        self.figure, self.axis = plt.subplots(2, 2, figsize=(15, 15))


        # learning_rate: or alpha | iterations: number of loop needed to train that fking model
        #                         | output: output file name to store final Theta0/Theta1
        self.l_rate = l_rate
        self.max_iter = 0
        #visualisation
            #po - original data
            #pn - normalized data
            #hs - history
            #live - live gif of training
        self.po = po
        self.pn = pn
        self.hs = hs
        self.live = live


def read_dataset(dataset_file):
    try:
        return pd.read_csv(dataset_file)
    except IOError:
        print("Error: could not read thetas' file")
        return None

def write_tethas(tethas):
    with open("thetas.txt", 'w') as file:
       file.write(f"{tethas[0]} {tethas[1]}") 

def estimate_price(mileage, thetas):
    return thetas[0] + mileage * thetas[1]

def normalise(thetas, mileage, mileage_scaled):
    price_predict_scaled = estimate_price(mileage_scaled, thetas)

    ml = [mileage[0], mileage[int(len(mileage) / 2) - 1]]
    pr = [price_predict_scaled[0], price_predict_scaled[int(len(price_predict_scaled) / 2) - 1]]
    
    thetas[1] = (pr[1] - pr[0]) / (ml[1] - ml[0])
    thetas[0] = pr[0] - thetas[1] * ml[0]
    return thetas

def condition_to_stop(self):
    if self.

def train(dataset, self):
    mileage = np.array(dataset['km'])
    price = np.array(dataset['price'])

    thetas = [0.0, 0.0]
    m = len(price)
    mileage_scaled = (mileage - np.mean(mileage)) / np.std(mileage)
    learning_rate = self.l_rate
    iter_nbr = 100

    with output(output_type='dict', sort_key=lambda x: 1) as output_lines:
        for _ in range(iter_nbr):
            tmp = [0.0, 0.0]
            tmp[0] = learning_rate / m * sum(estimate_price(mileage_scaled, thetas) - price)
            tmp[1] = learning_rate / m * sum((estimate_price(mileage_scaled, thetas) - price) * mileage_scaled)
            
            thetas[0] -= tmp[0]
            thetas[1] -= tmp[1]

    return normalise(thetas, mileage, mileage_scaled)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-in', action="store", dest="input", type=str, default='data.csv',
                        help='source of data file')
    parser.add_argument('--history', '-hs', action="store_true", dest="history", default=False,
                        help='save history to futur display')

    parser.add_argument('--original', '-po', action="store_true", dest="plot_original", default=False,
                        help="Enable to plot the original data sets")

    parser.add_argument('--normalized', '-pn', action="store_true", dest="plot_normalized", default=False,
                        help="Enable to plot the normalized data sets")

    parser.add_argument('--learning_rate', '-l', action="store", dest="rate", type=float, default=0.1,
                        help='Change learning coeficient. (default is 0.1)')

    parser.add_argument('--live', '-lv', action="store_true", dest="live", default=False,
                        help='Store live chnaged on gif graph')
    return parser.parse_args()

if __name__ == "__main__":

    parsed_args = parse_args()
    filename = parsed_args.input
    if not filename.endswith('.csv'):
        print("Wrong file format")
        sys.exit()

    dataset = read_dataset(filename)
    if dataset is None:
        sys.exit()
    if len(dataset) == 0:
        print("Dataset is empty")
        sys.exit()

    if (parsed_args.rate < 0.0000001 or parsed_args.rate > 1):
        parsed_args.rate = 0.1
    print("\033[33m{:s}\033[0m".format('Initial Params for training model:'))
    print(prCyan('    Learning Rate    : ') + str(parsed_args.rate))
    print(prCyan('    Plot Original    : ') + ('Enabled' if parsed_args.plot_original else 'Disabled'))
    print(prCyan('    Plot Normalized  : ') + ('Enabled' if parsed_args.plot_normalized else 'Disabled'))
    print(prCyan('    Plot History     : ') + ('Enabled' if parsed_args.history else 'Disabled'))
    print(prCyan('    DataSets File    : ') + parsed_args.input)


    lr = linear_regression(
        po = parsed_args.plot_original,
        pn = parsed_args.plot_normalized,
        hs = parsed_args.history,
        live = parsed_args.live,
        l_rate = parsed_args.rate)
    tethas = train(dataset)
    write_tethas(tethas if not tethas is None else [0, 0])



