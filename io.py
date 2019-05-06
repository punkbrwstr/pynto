import numpy as np
import pandas as pd

from pynto import Column


def read_frame(data_frame):
    def read_frame(stack):
        for header, col in data_frame.iteritems():
            def frame_col(date_range,col=col):
                return col.reindex(date_range).values
            stack.append(Column(header,f'read_frame {header}',frame_col))
    return read_frame

def read_csv(csv_file):
    frame = pd.read_csv(csv_file, index_col=0, header=None, parse_dates=True)
    return read_frame(frame)
