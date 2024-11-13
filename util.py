import sys

def prAll(skk): return "{}".format(skk)

def normalizeElem(list, elem):
    return ((elem - min(list)) / (max(list) - min(list)))


def denormalizeElem(list, elem):
    return ((elem * (max(list) - min(list))) + min(list))


def normalize(values):
    return [((v - min(values)) / (max(values) - min(values))) for v in values]


def get_gradient_csv(input):
    try:
        thetas = {}
        file = open(input, 'r')
        lines = file.readlines()
        for line in lines:
            thetas[line.strip().split(':')[0]] = float(line.strip().split(':')[1])
    except:
        thetas = {'T0': 0, 'T1': 0}
    return thetas


def set_gradient_csv(output, t0, t1):
    try:
        with open(output, "w+") as f:
            f.write('{} {}\n'.format(t0, t1))
    except:
        print("Cannot save results")
        sys.exit(0)
