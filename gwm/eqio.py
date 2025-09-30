import numpy as np
# -*- coding: utf-8 -*-
""" A function to read PEER NGA AT2 format

Created on Jan 08, 2024

@author: JS Nie @ US NRC
"""

def read_PEER_NGA_AT2(recfile):
    '''read PEER NGA AT2 acceleration time history'''
    assert recfile.endswith('.AT2'), f"{recfile} is not a PEER NGA AT2 file!"
    with open(recfile, 'r') as fh:
        header = fh.read(500) # NGA file header is around 300
        loc = header.find(', DT=')
        dt = float(header[loc+5:loc+13])
        locdata = header.find('\n', loc)
        # print(locdata, header[locdata:])
        fh.seek(0) # reset to the beginning of fh
        datastr = fh.read()[locdata:]
        data = np.fromstring(datastr, np.float32, sep=' ')
    return dt, data

