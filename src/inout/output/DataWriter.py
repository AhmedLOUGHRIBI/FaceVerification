from consts import Consts
from os.path import join
import pandas as pd
from datetime import datetime

class DataWriter():

    def write_as_csv(self, dictionary, csv_name, colname):
        df = pd.DataFrame.from_dict(dictionary, orient='index')
        df = df.reset_index().rename(columns={'index': 'key', 0: colname})
        path = join(Consts.output_directory, csv_name + ' ' +str(datetime.now().date()))
        print('Saving table {}'.format(csv_name))
        df.to_csv(path + '.csv', index=False)
        print('{} has been saved'.format(csv_name))