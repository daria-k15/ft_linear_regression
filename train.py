import sys
import matplotlib.pyplot as plt
import argparse
import imageio.v2 as imageio
import os
import csv
from reprint import output
from pandas import DataFrame
from util import *

class ft_linear_regression:
    def __init__(self, rate, dataset, output, plot_original, plot_normalized, plot_history, live, max_iter):
        self.figure, self.axis = plt.subplots(2, 2, figsize=(10,10))

        # X, Y normalized and x, y is orignal
        self.dataset = dataset
        self.x = dataset[0]
        self.y = dataset[1]
        self.X = normalize(self.x)
        self.Y = normalize(self.y)
        # M: Length of datasets | history_cost: history of cost | images: all iteration shots |
        #                       | MSE: Mean Square Error Percentage | RMSE: MSE**2 Percentage
        self.M = len(self.x)
        self.history_cost = []
        self.RMSE = None
        self.MSE = None

        # Theta normalized / _Theta denormalized
        self._T0 = 0
        self._T1 = 0
        self.T0 = 1
        self.T1 = 1

        # delta MSE to calculate the progress of accurracy
        self.prev_mse = 0.0
        self.cur_mse = self.cost()
        self.delta_mse = self.cost()

        # learning_rate: or alpha | iterations: number of loop needed to train that fking model
        #                         | output: output file name to store final Theta0/Theta1
        self.learning_rate = rate
        self.iterations = 0
        self.max_iterations = max_iter
        self.output = output

        # visualisation
        #   plot_original: plot original data
        #   plot_normalized: plot normalized data
        #   history: plot history of cost
        #   live: live watch of the training model
        self.po = plot_original
        self.pn = plot_normalized
        self.history = plot_history
        self.live = live

    def estimatePrice(self, t0, t1, mileage):
        return ((t0 + (t1 * float(mileage))))


    def RMSE_percent(self):
        self.RMSE = 100 * (1 - self.cost() ** 0.5)
        return self.RMSE

    def MSE_percent(self):
        self.MSE = 100 * (1 - self.cost())
        return self.MSE

    def predict(self, x):
        return self.T1 * x + self.T0

    def cost(self):
        dfX = DataFrame(self.X, columns=['X'])
        dfY = DataFrame(self.Y, columns=['Y'])
        return ((self.T1 * dfX['X'] + self.T0 - dfY['Y']) ** 2).sum() / self.M

    def live_update(self, output_lines):
        deltaX = max(self.x) - min(self.x)
        deltaY = max(self.y) - min(self.y)
        self._T1 = deltaY * self.T1 / deltaX
        self._T0 = ((deltaY * self.T0) + min(self.y) - self.T1 * (deltaY / deltaX) * min(self.x))
        output_lines[prAll('    Theta0           ')] = str(self.T0)
        output_lines[prAll('    Theta1           ')] = str(self.T1)
        output_lines[prAll('    RMSE             ')] = f'{round(self.RMSE_percent(), 2)} %'
        output_lines[prAll('    MSE              ')] = f'{round(self.MSE_percent(), 2)} %'
        output_lines[prAll('    Delta MSE        ')] = str(self.delta_mse)
        output_lines[prAll('    Iterations       ')] = str(self.iterations)
        
    def estimate_price(mileage, thetas):
        return thetas[0] + mileage * thetas[1]


    def gradient_descent(self):
        print('TRAINING MODEL :')
        with output(output_type='dict', sort_key=lambda x: 1) as output_lines:
            while self.iterations < self.max_iterations and abs(self.delta_mse) > 1e-7:
                sum1, sum2 = 0, 0
                for i in range(self.M):
                    error = self.predict(self.X[i]) - self.Y[i]
                    sum1 += error
                    sum2 += error * self.X[i]

                self.T0 -= self.learning_rate * sum1 / self.M
                self.T1 -= self.learning_rate * sum2 / self.M

                self.history_cost.append(self.cur_mse)
                self.prev_mse = self.cur_mse
                self.cur_mse = self.cost()
                self.delta_mse = self.cur_mse - self.prev_mse
                self.iterations += 1

                if self.iterations % 100 == 0 or self.iterations == 1:
                    self.live_update(output_lines)
                    if self.live == True:
                        self.plot_all(self)

            self.live_update(output_lines)
        self.RMSE_percent()
        self.MSE_percent()

        print('RESULTS (Normalized)  :')
        print(f'    {prAll("Theta0           :")} {self.T0}\n    {prAll("Theta1           :")} {self.T1}')
        print('RESULTS (DeNormalized):')
        print(f'    {prAll("Theta0           :")} {self._T0}\n    {prAll("Theta1           :")} {self._T1}')
        print('AlGORITHM ACCURACY:')
        print(f'    {prAll("RMSE             : ")}{round(linear_regression.RMSE, 2)} % ≈ ({linear_regression.RMSE} %)')
        print(f'    {prAll("MSE              : ")}{round(linear_regression.MSE, 2)} % ≈ ({linear_regression.MSE} %)')
        print(f'    {prAll("ΔMSE             : ")}{linear_regression.delta_mse}')
        print('Storing Theta0 && Theta1:')
        set_gradient_csv(self.output, self._T0, self._T1)
        print("    Theta0 && Theta1 has been stored in file , open : " + self.output)

        if self.po or self.pn or self.history:
            print('Plotting Data:')
            self.plot_all(final=True)
            print("    Data plotted successfully , open : " + 'LR-Graph.png')

        if self.live == True:
            print('Creating GIF image of progress:')
            self.gifit()
            print("    Live progress GIF created , open : " + 'LR-Live.gif')

    def gifit(self):
        if os.path.exists('./LR-Live.gif'):
            os.remove('./LR-Live.gif')
        def sorted_ls(path):
            mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
            return list(sorted(os.listdir(path), key=mtime))

        filenames = sorted_ls('./gif')
        with imageio.get_writer('./LR-Live.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread('./gif/' + filename)
                writer.append_data(image)

    def plot_original(self):
        p1 = self.axis[0, 0]
        p1.plot(self.x, self.y, 'ro', label='data')
        x_estim = self.x
        y_estim = [denormalizeElem(self.y, self.estimatePrice(self.T0, self.T1, normalizeElem(self.x, _))) for _ in
                   x_estim]
        p1.plot(x_estim, y_estim, 'g-', label='Estimation')
        p1.set_ylabel('Price (in euro)')
        p1.set_xlabel('Mileage (in km)')
        p1.set_title('Price = f(Mileage) | Original')

    def plot_normalized(self):
        p2 = self.axis[0, 1]
        p2.plot(self.X, self.Y, 'ro', label='data')
        x_estim = self.X
        y_estim = [self.estimatePrice(self.T0, self.T1, _) for _ in x_estim]
        p2.plot(x_estim, y_estim, 'g-', label='Estimation')

        p2.set_title('Price = f(Mileage) | Normalized')

    def plot_history(self):
        p4 = self.axis[1, 1]
        p4.set_ylabel('Cost')
        p4.set_xlabel('Iterations')
        p4.set_title(f'Cost = f(iteration) | L.Rate = {self.learning_rate}')
        p4.plot([i for i in range(self.iterations)], self.history_cost)

    def plot_show(self, final):
        if self.po != False or self.pn != False or self.history != False:
            if self.po == False:
                self.axis[0, 0].axis('off')

            if self.pn == False:
                self.axis[0, 1].axis('off')

            if self.history == False:
                self.axis[1, 1].axis('off')

            self.axis[1, 0].axis('off')

            # plt.show() # in case running from Pycharm or any other editors
            imgname = f'./gif/LR-Graph-{self.iterations}.png'
            if final == True:
                imgname = f'./LR-Graph.png'

            plt.savefig(imgname)
            plt.close()

    def plot_all(self,final=False):
        self.figure, self.axis = plt.subplots(2, 2, figsize=(10, 10))
        if self.po:
            self.plot_original()
        if self.pn:
            self.plot_normalized()
        if self.history:
            self.plot_history()

        self.plot_show(final)


def read_dataset(dataset_file):
    mileages = []
    prices = []
    with open(dataset_file, 'r') as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        for row in csvReader:
            mileages.append(row[0])
            prices.append(row[1])

    mileages.pop(0)
    prices.pop(0)
    for i in range(len(mileages)):
        mileages[i] = eval(mileages[i])
        prices[i] = eval(prices[i])
    return (mileages, prices)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-in', action="store", dest="input", type=str, default='data.csv')
    parser.add_argument('--output', '-out', action="store", dest="out", type=str, default='thetas.txt')
    parser.add_argument('--iteration', '-it', action="store", dest="iter", type=int, default=1000)
    parser.add_argument('--rate', '-r', action="store", dest="rate", type=float, default=0.1)
    parser.add_argument('--plot_original', '-po', action="store_true", dest="plot_original", default=False)
    parser.add_argument('--plot_normalized', '-pn', action="store_true", dest="plot_normalized", default=False)
    parser.add_argument('--live', '-lv', action="store_true", dest="live", default=False)
    parser.add_argument('--history', '-hs', action="store_true", dest="history", default=False)
    return parser.parse_args()


if __name__ == "__main__":

    if not os.path.exists('./gif'):
        os.makedirs('./gif')

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
    print('Initial Params for training model:')
    print('    Learning Rate    : ' + str(parsed_args.rate))
    print('    Plot Original    : ' + ('Enabled' if parsed_args.plot_original else 'Disabled'))
    print('    Plot Normalized  : ' + ('Enabled' if parsed_args.plot_normalized else 'Disabled'))
    print('    Plot History     : ' + ('Enabled' if parsed_args.history else 'Disabled'))
    print('    DataSets File    : ' + parsed_args.input)

    linear_regression = ft_linear_regression(
        plot_original = parsed_args.plot_original,
        plot_normalized = parsed_args.plot_normalized,
        plot_history = parsed_args.history,
        live = parsed_args.live,
        rate = parsed_args.rate,
        dataset = dataset,
        output = parsed_args.out,
        max_iter = parsed_args.iter)
    
    linear_regression.gradient_descent()