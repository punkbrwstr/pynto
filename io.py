import numpy as np
import pandas as pd


def read_frame(data_frame):
    def read_frame(stack):
        for header, col in data_frame.iteritems():
            def frame_col(date_range,col=col):
                return col.reindex(date_range).values
            stack.append((header,frame_col))
    return read_frame
