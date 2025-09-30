# -*- coding: utf-8 -*-
""" Utility codes to complement matplotlib

Created on Jan 08, 2024

@author: JS Nie @ US NRC
"""
import io
from time import time

import PIL.Image as Image
import PyQt5.QtWidgets
from matplotlib.offsetbox import DraggableAnnotation  
from matplotlib.text import Annotation
from matplotlib.widgets import SpanSelector  

PyQt5.QtWidgets.QApplication.setAttribute(
        PyQt5.QtCore.Qt.AA_EnableHighDpiScaling, True)


def savefig_reduced_png(fig, figname):
    # figname must be a png file
    strbuffer = io.BytesIO()
    fig.savefig(strbuffer)  # save to a memory file
    # the following line usually reduce the png file size to less than 25%
    Image.open(strbuffer).convert(mode='P').save(figname, optimize=True)
    strbuffer.close()


class mpl_iter:
    'provide a convenient indicator for loops'
    def __init__(self, fig, container, stoppable=False, unit='', 
                  mininterval=0.1,
                 ):
        self.fig = fig
        self.data = container
        # print('In mpl_iter:', self.data, type(self.data))
        self.stoppable = stoppable
        self.unit = unit
        self.mininterval = mininterval  # 0.1 second
        self.to_stop = False

    def __iter__(self):
        # initiate plot
        # matplotlib use point per inch (ppi) = 72
        if self.fig.canvas.widgetlock.locked():
            # return
            raise StopIteration("Figure canvas is locked! Uncheck Pan/Zoom etc. in the response spectrum matching window.")
        
        self.fig.canvas.widgetlock(self)
#        print(type(self.data), self.data)
        self.iter = iter(self.data)
        self.value = 0
        # make textsize relative to the smallest dimension of the figure
        # textsize = (min(self.fig.get_size_inches()) * self.fig.dpi / 20)
        self.txt = self.fig.text(0.5, 0.5,
                                 f'{self.value}/{len(self.data)}{self.unit}',
                                 size=40,
                                 fontweight='bold',
                                 family='monospace',
                                 color='darkgray',
                                 bbox=dict(facecolor='maroon',
                                           edgecolor='darkgray',
                                           boxstyle='circle',
                                           linewidth=5,
                                           alpha=0.9
                                           ),
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 animated=True,
                                 )
        if self.stoppable:
            self.click_cid = self.fig.canvas.mpl_connect(
                        'button_press_event',
                        self.onclick_to_stop)
        # set up update interval
        self.start_t = time()
        
        # prepare blit
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        # print(type(self.bg), dir(self.bg))
        self.fig.draw_artist(self.txt)
        self.fig.canvas.blit(self.fig.bbox)
        # bg1 = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        # print(type(bg1), dir(bg1))
        return self

    def __next__(self):
        try:
            if self.to_stop:
                raise StopIteration  # prematurely
            result = next(self.iter)
            self.value += 1
            cur_t = time()
            dt = cur_t - self.start_t
            if dt > self.mininterval or self.value == len(self.data):
                self.txt.set_text(f'{self.value}/{len(self.data)}{self.unit}')
                # blit new frame
                self.fig.canvas.restore_region(self.bg)
                self.fig.draw_artist(self.txt)
                self.fig.canvas.blit(self.fig.bbox)
                self.fig.canvas.flush_events()  # must have to make sure update
                self.start_t = cur_t
            # return the true item
            return result
        except StopIteration:
            # if self.fig.canvas.widgetlock.isowner(self):
            #     self.fig.canvas.widgetlock.release(self)
            self.txt.remove()
            self.fig.canvas.draw()
            raise StopIteration
        finally:
            if self.fig.canvas.widgetlock.isowner(self):
                    self.fig.canvas.widgetlock.release(self)

    def onclick_to_stop(self, event):
        self.fig.canvas.mpl_disconnect(self.click_cid)
        self.to_stop = True


def message_figure_annotation(fig, msg, x=0.5, y=0.5):
    'to show a message in fig using annotation'
    bbox = dict(facecolor='whitesmoke',
                edgecolor='xkcd:light maroon',  # 'eggplant',
                #  boxstyle='round,pad=1',
                boxstyle='roundtooth,pad=1',
                linewidth=2,
                alpha=0.9
                )
    ann = Annotation(msg, (x, y), xytext=None, 
                    xycoords='figure fraction', 
                    textcoords=None,
                    family='monospace',
                    bbox=bbox,
                    horizontalalignment='center',  # 'left',
                    verticalalignment='center',
                    )
    fig.add_artist(ann) # this needs to be before add DraggableAnnotation
    ann.__draggable = DraggableAnnotation(ann, use_blit=True)
    return ann


class LockableSpanSelector(SpanSelector):
    'avoid cursor drawing'
    def press(self, event):
        if self.canvas.widgetlock.locked():
            return
        self.canvas.widgetlock(self)
        SpanSelector.press(self, event)

    def release(self, event):
        if self.canvas.widgetlock.isowner(self):
            self.canvas.widgetlock.release(self)
            SpanSelector.release(self, event)

