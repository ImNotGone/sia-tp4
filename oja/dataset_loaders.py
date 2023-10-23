import csv
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
import scipy.stats as stats

def load_countries(file_path: str) -> Tuple[List[str], List[str], NDArray]:
    country_data = []
    countries = []
    variables = []

    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        variables = next(csv_reader)[1:]

        for row in csv_reader:
            countries.append(row[0])

            # Convert strings to floats
            country_data.append([float(x) for x in row[1:]])

    dataset = np.array(country_data)
    
    return countries, variables, dataset
