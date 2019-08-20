import numpy as np
import pandas as pd
from pynto.main import Column
from pynto.time import Range

def pandas_frame(data_frame, header=0):
    def read_frame(stack):
        for header, col in data_frame.iteritems():
            def frame_col(date_range,col=col):
                return col.reindex(date_range.to_pandas(), method='ffill').values
            stack.append(Column(header,f'read_frame {header}',frame_col))
    return read_frame

def csv(csv_file, header=0):
    frame = pd.read_csv(csv_file, index_col=0, header=header, parse_dates=True)
    return pandas_frame(frame)
