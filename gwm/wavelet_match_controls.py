# -*- coding: utf-8 -*-
""" GUI for GWM - Greedy Wavelet Method for response spectrum matching

 Jan 08, 2024

@author: JS Nie @ US NRC
"""
from pathlib import Path
from functools import partial

# PyQt6 imports (commented out for now - uncomment to use PyQt6)
# from PyQt6.QtWidgets import (QWidget, QPushButton, QGridLayout,
#                              QShortcut, QFrame, QInputDialog,
#                              QGroupBox, QVBoxLayout, QSplashScreen )
# from PyQt6.QtCore import Qt, QTimer
# from PyQt6.QtGui import QKeySequence, QIcon, QPixmap

# PyQt5 imports (current)
from PyQt5.QtWidgets import (QWidget, QPushButton, QGridLayout,
                             QShortcut, QFrame, QInputDialog,
                             QGroupBox, QVBoxLayout, QSplashScreen )
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence, QIcon, QPixmap

# button_style = """
# QPushButton {
#     background-color: #f0f0f0;
#     border: 1px solid #c0c0c0;
#     border-radius: 4px;
#     padding: 5px 10px;
#     font-weight: bold;
# }
# QPushButton:hover {
#     background-color: #e0e0e0;
# }
# QPushButton:pressed {
#     background-color: #d0d0d0;
# }
# QPushButton:checked {
#     background-color: #4CAF50;
#     color: white;
# }
# """


class WM_Controls(QWidget):
   
    def __init__(self, wmparent, name_actions=[]):
        self.wm = wmparent
        # self.options['AutoSaveInverval'] = 20
        if wmparent:
            self.parent = wmparent.fig.canvas.manager.window
        self.name_actions = name_actions # (name, do)
        super().__init__()
        self.icon_file = str(Path(__file__).parent / 'resources/gwm_logo.svg')
        # self.flashSplash()
        self.initUI()

    def initUI(self):   
        self.gwm_icon = QIcon(self.icon_file)
        self.setWindowIcon(self.gwm_icon)
        self.states = {}
        grid = QGridLayout()  
        self.setLayout(grid)
        # i for row, j for column
        # j = 0
        for grp in self.name_actions:
            grpname, r, c = grp[0]
            grpbox = QGroupBox(grpname)
            vbox = QVBoxLayout()
            for name, action, *args in grp[1:]:
                if name == '--':
                    button = QFrame()
                    button.setFrameShape(QFrame.HLine)
                    button.setLineWidth(1)
                # button.setStyleSheet("background-color: orange;")
                # elif isinstance(action, int): 
                #     button = QPushButton(name + f':{action}')
                #     self.maxiter = action
                #     self.states[name] = button
                #     button.clicked.connect(partial(self.get_maxiter, name, button))                
                else:
                    button = QPushButton(name)
                    button.clicked.connect(action)
                    if args:
                        button.setCheckable(True)
                        button.setChecked(args[0])
                        self.states[name] = button
                vbox.addWidget(button)
                # button.setStyleSheet(button_style)
            vbox.addStretch()
            grpbox.setLayout(vbox)
            grid.addWidget(grpbox, r, c)
            # button.setMaximumSize(button.sizeHint())
            # grid.addWidget(button, i, j, 1, 1) 

        # grid.setColumnStretch(0, 1)
        # grid.setColumnMinimumWidth(0, 30)
        self.setFixedHeight(250)
        self.set_location_relative_to_parent()
        self.set_title()
        # self.setWindowOpacity(0.9)
        # self.setWindowFlags(Qt.CustomizeWindowHint 
        #                     | Qt.WindowCloseButtonHint
        #                     | Qt.WindowStaysOnTopHint)
        self.shortcut_hide = QShortcut(QKeySequence('F1'), self)
        self.shortcut_hide.activated.connect(self.on_f1_hide)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet('background-color: whitesmoke;')

        # szpolicy = self.sizePolicy()
        # szpolicy.setHorizontalPolicy(QSizePolicy.Expanding )
        self.show()
    
    def flashSplash(self):
        self.splash = QSplashScreen(QPixmap(self.icon_file).scaledToHeight(500),
                                    Qt.WindowStaysOnTopHint)
        # By default, SplashScreen will be in the center of the screen.
        # You can move it to a specific location if you want:
        # self.splash.move(10,10)
        self.splash.show()
        # Close SplashScreen after 2 seconds (2000 ms)
        QTimer.singleShot(2000, self.splash.close)
        
    # def get_maxiter(self, name, button):
    #     integer, done = QInputDialog.getInt(
    #         self, 'Input Dialog', 'Enter an integer:',
    #         self.maxiter)
    #     if done:
    #         self.maxiter = integer
    #         button.setText(f'{name}:{integer}')
    
    def get_state(self, button_name):
        return self.states[button_name].isChecked()
    
    def on_f1_hide(self):
        self.hide()
        self.wm.fig.canvas.setFocus()
    
    def set_location_relative_to_parent(self):
        # top = self.parent.y() + self.parent.height() // 3
        # left = self.parent.x()
        top = self.wm.screen_height - self.height() - 61
        left = self.parent.x() + self.parent.width() + 1
        self.move(left, top)

    def closeEvent(self, event):
        self.wm.on_close(None)
        event.accept()

    def set_title(self, titlemsg='Wavelet Matching Controls'):
        self.setWindowTitle(titlemsg) 