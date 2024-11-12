import sys
import numpy as np
import pandas as pd

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

def train(dataset):
    mileage = np.array(dataset['km'])
    price = np.array(dataset['price'])

    thetas = [0.0, 0.0]
    m = len(price)
    mileage_scaled = (mileage - np.mean(mileage)) / np.std(mileage)
    learning_rate = 0.2
    iter_nbr = 100

    for _ in range(iter_nbr):
        tmp = [0.0, 0.0]
        tmp[0] = learning_rate / m * sum(estimate_price(mileage_scaled, thetas) - price)
        tmp[1] = learning_rate / m * sum((estimate_price(mileage_scaled, thetas) - price) * mileage_scaled)
        
        thetas[0] -= tmp[0]
        thetas[1] -= tmp[1]

    return normalise(thetas, mileage, mileage_scaled)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        if not filename.endswith('.csv'):
            print("Wrong file format")
            sys.exit()
        dataset = read_dataset(filename)
        if dataset is None:
            sys.exit()
        if len(dataset) == 0:
            print("Dataset is empty")
            sys.exit()
        tethas = train(dataset)
        write_tethas(tethas if not tethas is None else [0, 0])
    else:
        print("Wrong number of arguments")