# -*- coding: utf-8 -*-
""" A couple classes to achieve draggable/moveable objects

Created on Jan 08, 2024

@author: JS Nie @ US NRC
"""
# A draggable or editable artist can be defined easily following the
# examples: DraggableRectangle, DraggableVline, and EditableLine
# based on draggable rectangle with the animation blit techniques; see
# http://www.scipy.org/Cookbook/Matplotlib/Animations
import numpy as np
from matplotlib.transforms import Affine2D


def _move_log(x0, xpress, xdata):
    return 10.0**(np.log10(x0) + np.log10(xdata) - np.log10(xpress))


def _move_linear(x0, xpress, xdata):
    return x0 + xdata - xpress


class Draggable(object):
    lock = None  # only one can be animated at a time
    def __init__(self, artist, user_draw=None, connect=True, get_data=None,
                 move_x=True, move_y=True):
        self.artist = artist
        self.user_draw = user_draw
        self.get_data = get_data
        self.move_x = move_x
        self.move_y = move_y
        self.data2axes = artist.axes.transScale + artist.axes.transLimits
        self.clean()
        if connect:
            self.connect()

    def picked(self, event):
        'test if event picked the artist, must be implemented'
        contains, attrd = self.artist.contains(event)
        # if contains:
        #     print(self.__class__.__name__)
        return contains

    def init_state(self, event):
        'initial state for motion'
        if self.move_x:
            if self.artist.axes.get_xscale() == 'log':
                self.newx = _move_log
            else:
                self.newx = _move_linear
        else:
            self.newx = None

        if self.move_y:
            if self.artist.axes.get_yscale() == 'log':
                self.newy = _move_log
            else:
                self.newy = _move_linear
        else:
            self.newy = None

    def move(self, event):
        'move to event'
        raise NotImplementedError('Must be impleted')

    def clean(self):
        '''clean/initialize instant variables on release

        pick: return(s) of self.picked (bool or index or any other user specified)
        press: return value(s) of self.init_state
        bacground: private variable used by Draggable, not supposed to be changed.
        '''
        self.press = self.pick = self.background = None

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.artist.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.artist.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.artist.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        return self

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if not self.artist.get_visible(): return
        if event.inaxes != self.artist.axes: return
        if Draggable.lock is not None: return
        self.pick = self.picked(event)
        if not self.pick:
            return

        #~ print('event contains', self.rect.xy)
        self.init_state(event)
        Draggable.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.artist.figure.canvas
        axes = self.artist.axes
        self.artist.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.artist.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.artist)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)


    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if Draggable.lock is not self:
            return
        if event.inaxes != self.artist.axes: return
        self.move(event)
        canvas = self.artist.figure.canvas
        axes = self.artist.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.artist)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        'on release we reset the press data'
        if Draggable.lock is not self:
            return

        Draggable.lock = None

        # turn off the rect animation property and reset the background
        self.artist.set_animated(False)
        #~ self.background = None
        self.clean()

        # trigger a user_draw event
        if self.user_draw:
            self.user_draw(self.artist)

        # redraw the full figure
        self.artist.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.artist.figure.canvas.mpl_disconnect(self.cidpress)
        self.artist.figure.canvas.mpl_disconnect(self.cidrelease)
        self.artist.figure.canvas.mpl_disconnect(self.cidmotion)


class Movable(Draggable):
    'works for arists with get_position and set_position'
    def init_state(self, event):
        'initial state for motion, called in on_press'
        super().init_state(event)
        x0, y0 = self.artist.get_position()
        self.press = x0, y0, event.xdata, event.ydata

    def move(self, event):
        'move to event'
        x0, y0, xpress, ypress = self.press
        if self.newx:
            self.artist.set_x(self.newx(x0, xpress, event.xdata))
        if self.newy:
            self.artist.set_y(self.newy(y0, ypress, event.ydata))
        _move(self.artist, self.press, event.xdata, event.ydata)


DraggableText = MovableText = MovableRectangle = DraggableRectangle = Movable

class DraggableLine2D(Draggable):
    def init_state(self, event):
        'initial state for motion'
        super().init_state(event)
        x0 = self.artist.get_xdata()
        y0 = self.artist.get_ydata()
        self.press = x0, y0, event.xdata, event.ydata

    def move(self, event):
        'move to event'
        x0, y0, xpress, ypress = self.press
        if self.newx:
            self.artist.set_xdata(self.newx(x0, xpress, event.xdata))
        if self.newy:
            self.artist.set_ydata(self.newy(y0, ypress, event.ydata))

