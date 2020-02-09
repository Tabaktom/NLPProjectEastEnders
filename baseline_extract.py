import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
class baseline():
    def gender(y_vector):
        predictions = []
        for y in y_vector:
            rand = np.random.randint(low = 0,high =2, size = 1)
            if rand == 1:
                pred = 'male'
            else:
                pred = 'female'
            predictions.append(pred)
        report = classification_report(y_vector, pd.Series(predictions))
        return report

    def characters(y_vector, characters_list):
        predictions =[]
        for ind, y in enumerate(y_vector):
            rand = np.random.randint(low = 0,high =len(characters_list), size = 1)[0]
            char = characters_list[rand]
            predictions.append(char)
        report = classification_report(y_vector, pd.Series(predictions))
        return report