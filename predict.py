import re
import os
import sys

def read_thetas():
    try:
        with open("thetas.txt", 'r') as file:
            thetas = file.read()
            if not re.match(r"[+-]?(\d*[.])?\d+", thetas):
                print("File contains wrong data")
                return None
            return [float(theta) for theta in thetas.split(' ')]
    except IOError:
        print("Error: could not read thetas' file")
        return None

def predict(milage, thetas):
    return thetas[0] + milage * thetas[1]

def delete_thetas_file():
    if (os.path.exists("thetas.txt")):
        os.remove("thetas.txt")

if __name__ == "__main__":
    thetas = read_thetas()
    if thetas is None:
        sys.exit()
    while True:
        mileage = input("Enter mileage: ")
        if (mileage == "q"):
           break
        print("Mileage must be an integer" if not mileage.isdigit() else f"Estimate price: {predict(int(mileage), thetas)}")
    
    delete_thetas_file()
