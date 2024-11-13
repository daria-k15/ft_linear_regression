import sys
import matplotlib.pyplot as plt
import argparse
import imageio.v2 as imageio
import os
from reprint import output
from util import *

class ft_linear_regression:
    def __init__(self, rate, dataset, output, po, pn, history, live, iter):
        self.figure, self.axis = plt.subplots(2, 2, figsize=(15,15))

        # X, Y normalized and x, y is orignal
        self.dataset = dataset
        self.x = dataset[0]
        self.y = dataset[1]
        self.X = normalisation(self.x)
        self.Y = normalisation(self.y)
        # M: Length of datasets | C: history of cost | images: all iteration shots |
        #                       | MSE: Mean Square Error Percentage | RMSE: MSE**2 Percentage
        self.M = len(self.x)
        self.C = []
        self.images = []
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
        self.delta_mse = self.cur_mse

        # learning_rate: or alpha | iterations: number of loop needed to train that fking model
        #                         | output: output file name to store final Theta0/Theta1
        self.learning_rate = rate
        self.iterations = 0
        self.max_iterations = iter
        self.output = output

        # visualisation
        #   po: plot original data
        #   pn: plot normalized data
        #   history: plot history of cost
        #   live: live watch of the training model
        self.po = po
        self.pn = pn
        self.history = history
        self.live = live

    def MSE_percent(self):
        self.MSE = 100 * self.cost()
        return self.MSE

    def estimatePrice(self, t0, t1, mileage):
        return ((t0 + (t1 * float(mileage))))


    def RMSE_percent(self):
        self.RMSE = 100 * (1 - self.cost() ** 0.5)
        return self.RMSE

    #MSE
#     self.cost() — вызывает функцию cost, чтобы получить MSE.
# self.cost() ** 0.5 — вычисляет корень квадратный из MSE, получая RMSE.
# 1 - self.cost() ** 0.5 — вычитает RMSE из 1. Это может быть полезно для оценки "качества" модели: если модель идеально предсказывает, то RMSE будет равно 0, и результат будет равен 1. То есть ошибка минимальна.
# 100 * (1 - self.cost() ** 0.5) — умножает результат на 100, чтобы получить процентное значение ошибки.
    def cost(self):
        dfX = DataFrame(self.X, columns=['X'])
        dfY = DataFrame(self.Y, columns=['Y'])
        return ((self.T1 * dfX['X'] + self.T0 - dfY['Y']) **2).sum() / self.M

    def live_update(self, output_lines):
        deltaX = max(self.x) - min(self.x)
        deltaY = max(self.y) - min(self.y)
        self._T1 = deltaY * self.T1 / deltaX
        self._T0 = ((deltaY * self.T0) + min(self.y) - self.T1 * (deltaY / deltaX) * min(self.x))
        output_lines[prCyan('    Theta0           ')] = str(self.T0)
        output_lines[prCyan('    Theta1           ')] = str(self.T1)
        output_lines[prCyan('    RMSE             ')] = f'{round(self.RMSE_percent(), 2)} %'
        output_lines[prCyan('    MSE              ')] = f'{round(self.MSE_percent(), 2)} %'
        output_lines[prCyan('    Delta MSE        ')] = str(self.delta_mse)
        output_lines[prCyan('    Iterations       ')] = str(self.iterations)

    def condition_to_stop_training(self):
        if self.max_iterations == 0:
            return self.delta_mse > 0.0000001 or self.delta_mse < -0.0000001
        else:
            return self.iterations < self.max_iterations
        
    def estimate_price(mileage, thetas):
        return thetas[0] + mileage * thetas[1]


    def gradient_descent(self):
        print("\033[33m{:s}\033[0m".format('TRAINING MODEL :'))
        self.iterations = 0
        with output(output_type='dict', sort_key=lambda x: 1) as output_lines:
            while self.condition_to_stop_training():
                tmp = [0.0, 0.0]
                for i in range(self.M):
                    T = self.T0 + self.T1 * self.X[i] - self.Y[i]
                    tmp[0] += T
                    tmp[1] += T * self.X[i]

                self.T0 -= self.learning_rate * (tmp[0] / self.M)
                self.T1 -= self.learning_rate * (tmp[1] / self.M)

                self.C.append(self.cost())

                self.prev_mse = self.cur_mse
                self.cur_mse = self.cost()
                self.delta_mse = self.cur_mse - self.prev_mse

                self.iterations += 1

                if self.iterations % 100 == 0 or self.iterations == 1:
                    self.live_update(output_lines)
                    if self.live == True:
                        self.plot_all(self.po, self.pn, self.history)

            self.live_update(output_lines)
        self.RMSE_percent()
        self.MSE_percent()

        print(prYellow('SUCCESS :'))
        print(prGreen("    Applied model to data"))
        print(prYellow('RESULTS (Normalized)  :'))
        print(f'    {prCyan("Theta0           :")} {self.T0}\n    {prCyan("Theta1           :")} {self.T1}')
        print(prYellow('RESULTS (DeNormalized):'))
        print(f'    {prCyan("Theta0           :")} {self._T0}\n    {prCyan("Theta1           :")} {self._T1}')
        print("\033[33m{:s}\033[0m".format('AlGORITHM ACCURACY:'))
        print(f'    {prCyan("RMSE             : ")}{round(linear_regression.RMSE, 2)} % ≈ ({linear_regression.RMSE} %)')
        print(f'    {prCyan("MSE              : ")}{round(linear_regression.MSE, 2)} % ≈ ({linear_regression.MSE} %)')
        print(f'    {prCyan("ΔMSE             : ")}{linear_regression.delta_mse}')
        print(prYellow('Storing Theta0 && Theta1:'))
        set_gradient_csv(self.output, self._T0, self._T1)
        print(prGreen("    Theta0 && Theta1 has been stored in file , open : ") + self.output)

        if self.po or self.pn or self.history:
            print(prYellow('Plotting Data:'))
            self.plot_all(self.po, self.pn, self.history, final=True)
            print(prGreen("    Data plotted successfully , open : ") + 'LR-Graph.png')

        if self.live == True:
            print(prYellow('Creating GIF image of progress:'))
            self.gifit()
            print(prGreen("    Live progress GIF created , open : ") + 'LR-Live.gif')

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
        p4.plot([i for i in range(self.iterations)], self.C)

    def plot_show(self, p1, p2, p4, final):
        if p1 != False or p2 != False or p4 != False:
            if p1 == False:
                self.axis[0, 0].axis('off')

            if p2 == False:
                self.axis[0, 1].axis('off')

            if p4 == False:
                self.axis[1, 1].axis('off')

            self.axis[1, 0].axis('off')

            # plt.show() # in case running from Pycharm or any other editors
            imgname = f'./gif/LR-Graph-{self.iterations}.png'
            if final == True:
                imgname = f'./LR-Graph.png'

            plt.savefig(imgname)
            plt.close()

    def plot_all(self, p1, p2, p4, final=False):

        self.figure, self.axis = plt.subplots(2, 2, figsize=(10, 10))

        if p1:
            self.plot_original()
        if p2:
            self.plot_normalized()
        if p4:
            self.plot_history()

        self.plot_show(p1, p2, p4, final)


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
    parser.add_argument('--history', '-hs', action="store_true", dest="history", default=False)
    parser.add_argument('--original', '-po', action="store_true", dest="plot_original", default=False,
                        help="Enable to plot the original data sets")

    parser.add_argument('--normalized', '-pn', action="store_true", dest="plot_normalized", default=False,
                        help="Enable to plot the normalized data sets")

    parser.add_argument('--learning_rate', '-l', action="store", dest="rate", type=float, default=0.1,
                        help='Change learning coeficient. (default is 0.1)')

    parser.add_argument('--live', '-lv', action="store_true", dest="live", default=False,
                        help='Store live chnaged on gif graph')
    
    parser.add_argument('--iteration', '-it', action="store", dest="iter", type=int, default=0,
                        help='Change number of iteration. (default is Uncapped)')
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
    print("\033[33m{:s}\033[0m".format('Initial Params for training model:'))
    print(prCyan('    Learning Rate    : ') + str(parsed_args.rate))
    print(prCyan('    Plot Original    : ') + ('Enabled' if parsed_args.plot_original else 'Disabled'))
    print(prCyan('    Plot Normalized  : ') + ('Enabled' if parsed_args.plot_normalized else 'Disabled'))
    print(prCyan('    Plot History     : ') + ('Enabled' if parsed_args.history else 'Disabled'))
    print(prCyan('    DataSets File    : ') + parsed_args.input)

    linear_regression = ft_linear_regression(
        po = parsed_args.plot_original,
        pn = parsed_args.plot_normalized,
        history = parsed_args.history,
        live = parsed_args.live,
        rate = parsed_args.rate,
        dataset = dataset,
        output = parsed_args.out,
        iter = parsed_args.iter)
    
    linear_regression.gradient_descent()