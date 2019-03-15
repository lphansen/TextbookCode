#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the code for the Jupyter widgets. It is not required
for the model framework. The widgets are purely for decorative purposes.
"""

#######################################################
#                    Dependencies                     #
#######################################################

from ipywidgets import widgets, Layout, Button, HBox, VBox, interactive
from IPython.core.display import display
from IPython.display import clear_output, Markdown, Latex
from collections import OrderedDict
from IPython.display import Javascript
import json
import itertools
import numpy as np
from model_code_2d import *
import os
try:
    import plotly.graph_objs as go
    from plotly.tools import make_subplots
    from plotly.offline import init_notebook_mode, iplot
except ImportError:
    print("Installing plotly. This may take a while.")
    from pip._internal import main as pipmain
    pipmain(['install', 'plotly'])
    import plotly.graph_objs as go
    from plotly.tools import make_subplots
    from plotly.offline import init_notebook_mode, iplot

# Define global parameters for parameter checks
params_pass = False
model_solved = False

#######################################################
#          Jupyter widgets for user inputs            #
#######################################################

## This section creates the widgets that will be diplayed and used by the user
## to input parameter values.

style_mini = {'description_width': '5px'}
style_short = {'description_width': '100px'}
style_med = {'description_width': '200px'}
style_long = {'description_width': '200px'}

layout_mini =Layout(width='18.75%')
layout_50 =Layout(width='50%')
layout_med =Layout(width='70%')

widget_layout = Layout(width = '100%')

A = widgets.BoundedFloatText( ## fraction of new borns
    value=0.0355,
    min = 0,
    max = 2,
    step=0.0001,
    disabled=False,
    description = 'Productivity',
    style=style_med,
    layout = Layout(width='70%')
)
alpha_k = widgets.BoundedFloatText( ## death rate
    value=0.025,
    min = 0,
    max = 1,
    step=0.0001,
    disabled=False,
    description = 'Depreciation',
    style = style_med,
    layout = Layout(width='70%')
)
phi1 = widgets.BoundedFloatText( ## death rate
    value=0.0125,
    min = 0.00001,
    max = 10000,
    step=0.00001,
    disabled=False,
    description = '$\phi_1$',
    style = style_med,
    layout = Layout(width='70%')
)
phi2 = widgets.BoundedFloatText( ## death rate
    value=400,
    min = 0.00001,
    max = 10000,
    step=0.00001,
    disabled=False,
    description = '$\phi_2$',
    style = style_med,
    layout = Layout(width='70%')
)
beta_1 = widgets.BoundedFloatText( ## death rate
    value=0.014,
    min = 0,
    max = 1,
    step=0.0001,
    disabled=False,
    description = r'Technology shock ($\beta_1$)',
    style = style_med,
    layout = Layout(width='70%')
)
beta_2 = widgets.BoundedFloatText( ## death rate
    value=0.0022,
    min = 0,
    max = 10,
    step=0.0001,
    disabled=False,
    description = r'Preferences shock ($\beta_2$)',
    style = style_med,
    layout = Layout(width='70%')
)
delta = widgets.BoundedFloatText( ## death rate
    value=0.005,
    min = 0,
    max = 10,
    step=0.0001,
    disabled=False,
    description = 'Rate of Time Preference',
    style = style_med,
    layout = Layout(width='70%')
)
rhos = widgets.Text( ## death rate
    value="0.5, 1, 2",
    disabled=False,
    description = 'Inverse IES',
    style = style_med,
    layout = Layout(width='70%')
)

B11 = widgets.BoundedFloatText( ## cov11
    value= 0.011,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
B12 = widgets.BoundedFloatText( ## cov11
    value= 0.025,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
B13 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
B14 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
B21 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
B22 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
B23 = widgets.BoundedFloatText( ## cov11
    value= 0.119,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
B24 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
sigk1 = widgets.BoundedFloatText( ## cov11
    value= 0.477,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
sigk2 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
sigk3 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
sigk4 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
sigd1 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
sigd2 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
sigd3 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
sigd4 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
shock = widgets.BoundedIntText( ## death rate
    value=1,
    min = 1,
    max = 4,
    step=1,
    disabled=False,
    description = 'Shock Number',
    style = {'description_width': '180px'},
    layout = Layout(width='70%')
)
timeHorizon = widgets.BoundedIntText( ## death rate
    value=160,
    min = 10,
    max = 2000,
    step=10,
    disabled=False,
    description = 'Time Horizon (quarters)',
    style = {'description_width': '180px'},
    layout = Layout(width='70%')
)
folderName = widgets.Text(
    value='defaultModel',
    placeholder='defaultModel',
    description='Folder name',
    disabled=False,
    style = {'description_width': '180px'},
    layout = Layout(width='70%')
)

gammaSave = widgets.BoundedFloatText( ## death rate
    value=5,
    min = 1,
    max = 10,
    step=1,
    disabled=False,
    description = r'Risk aversion ($\gamma$) for saved plot',
    style = {'description_width': '180px'},
    layout = Layout(width='70%')
)

plotName = widgets.Text(
    value='Stochastic Growth',
    placeholder='Stochastic Growth',
    description='Plot Title',
    disabled=False,
    style = {'description_width': '180px'},
    layout = Layout(width='70%')
)

overwrite = widgets.Dropdown(
    options = {'Yes', 'No'},
    value = 'Yes',
    description='Overwrite if folder exists:',
    disabled=False,
    style = {'description_width': '180px'},
    layout = Layout(width='70%')
)

checkParams = widgets.Button(
    description='Update parameters',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

runModel = widgets.Button(
    description='Run model',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

dumpPlotPanel = widgets.Button(
    description='Save panel chart',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

showSS = widgets.Button(
    description='Show steady states',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

displayPlotPanel = widgets.Button(
    description='Show panel chart',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

box_layout       = Layout(width='100%', flex_flow = 'row')#, justify_content='space-between')
box_layout_wide  = Layout(width='100%', justify_content='space-between')
box_layout_small = Layout(width='10%')

Economy_box = VBox([widgets.Label(value="Economy"), A, alpha_k], \
layout = Layout(width='90%'))
Adjustment_box = VBox([widgets.Label(value="Adjustment Costs"), phi1, phi2], \
layout = Layout(width='90%'))
Preferences_box = VBox([widgets.Label(value="Preference parameters"), rhos, delta], \
layout = Layout(width='90%'))
Persistence_box = VBox([widgets.Label(value="Shock persistence"), beta_1, beta_2], \
layout = Layout(width='90%'))

B_box1 = HBox([B11, B12, B13, B14], layout = Layout(width='90%'))
B_box2 = HBox([B21, B22, B23, B24], layout = Layout(width='90%'))
B_box = VBox([widgets.Label(value="B matrix"),B_box1, B_box2], \
             layout = Layout(width='100%'))

sigk_box1 = HBox([sigk1, sigk2, sigk3, sigk4], layout = Layout(width='100%'))
sigk_box = VBox([widgets.Label(value="Sigma k"), sigk_box1], layout = Layout(width='90%'))
sigd_box1 = HBox([sigd1, sigd2, sigd3, sigd4], layout = Layout(width='100%'))
sigd_box = VBox([widgets.Label(value="Sigma d"), sigd_box1], layout = Layout(width='90%'))

sigmas_box = VBox([sigk_box, sigd_box], layout = Layout(width='100%'))

Selector_box = VBox([widgets.Label(value="Graph parameters"), shock, plotName, timeHorizon], \
                    layout = Layout(width='90%'))
System_box = VBox([widgets.Label(value="System parameters"), folderName, overwrite, gammaSave], \
                  layout = Layout(width='90%'))



line1      = HBox([Economy_box, Adjustment_box], layout = box_layout)
line2      = HBox([Preferences_box, Persistence_box], layout = box_layout)
line3      = HBox([B_box, sigmas_box], layout = box_layout)
line6      = HBox([Selector_box, System_box], layout = box_layout)
paramsPanel = VBox([line1, line2, line3, line6])
run_box = VBox([widgets.Label(value="Execute Model"), checkParams, runModel, \
                showSS, displayPlotPanel, dumpPlotPanel])



#######################################################
#                      Functions                      #
#######################################################

def checkParamsFn(b):
    ## This is the function triggered by the updateParams button. It will
    ## check dictionary params to ensure that adjustment costs are well-specified.
    clear_output() ## clear the output of the existing print-out
    display(run_box) ## after clearing output, re-display buttons
    global params_pass
    global model_solved
    model_solved = False
    if phi1.value * phi2.value < .6:
        params_pass = False
        print("Given your parameter values, phi 1 must be greater than {}.".format(round(.6 / phi2.value, 3)))
    else:
        rho_vals = np.array([np.float(r) for r in rhos.value.split(',')])
        rho_min = np.min(rho_vals)
        rho_max = np.max(rho_vals)
        upper_cutoff = (delta.value / (1 - rho_min) + alpha_k.value) / \
                        np.log(1 + phi2.value * A.value)
        lower_cutoff = (delta.value / (1 - rho_max) + alpha_k.value) / \
                        np.log(1 + phi2.value * A.value)
        lower_cutoff = min(lower_cutoff, 0.6 / phi2.value)
        if rho_min < 1 and rho_max > 1:
            if phi1.value > upper_cutoff:
                params_pass = False
                print("Given your parameter values, phi 1 must be greater than {} and less than {}.".format(round(lower_cutoff, 3), round(upper_cutoff, 3)))
            elif phi1.value < lower_cutoff:
                params_pass = False
                print("Given your parameter values, phi 1 must be greater than {} and less than {}.".format(round(lower_cutoff, 3), round(upper_cutoff, 3)))
            else:
                params_pass = True
                print("Parameter check passed.")
        elif rho_max > 1:
            if phi1.value < lower_cutoff:
                params_pass = False
                print("Given your parameter values, phi 1 must be greater than {}.".format(round(lower_cutoff, 3)))
            else:
                params_pass = True
                print("Parameter check passed.")
        elif rho_max < 1:
            if phi1.value > upper_cutoff:
                params_pass = False
                print("Given your parameter values, phi 1 must be less than {}.".format(round(upper_cutoff, 3)))
            else:
                params_pass = True
                print("Parameter check passed.")
        else:
            params_pass = True
            print("Parameter check passed.")

def runModelFn(b):
    ## This is the function triggered by the runModel button.
    global model_solved
    if params_pass:
        print("Solving the model...")
        display(Javascript("Jupyter.notebook.execute_cells([9])"))
        model_solved = True
        print("Done.")
    else:
        print("You must update the parameters first.")

def showSSFn(b):
    if model_solved:
        print("Showing steady state values.")
        display(Javascript("Jupyter.notebook.execute_cells([10])"))
    else:
        print("You must run the model first.")

def displayPlotPanelFn(b):
    if model_solved:
        print("Showing plots.")
        display(Javascript("Jupyter.notebook.execute_cells([11])"))
    else:
        print("You must run the model first.")

def dumpPlotPanelFn(b):
    if params_pass:
        if overwrite.value == "No" and os.path.isdir(folderName.value):
            print("Folder already exists and overwrite is set to 'No'. Change folder name or set overwrite to 'Yes'.")
        else:
            display(Javascript("Jupyter.notebook.execute_cells([12])"))
    else:
        print("You must update the parameters first.")


#######################################################
#                 Configure buttons                   #
#######################################################

selectedMoments = []

checkParams.on_click(checkParamsFn)
runModel.on_click(runModelFn)
showSS.on_click(showSSFn)
displayPlotPanel.on_click(displayPlotPanelFn)
dumpPlotPanel.on_click(dumpPlotPanelFn)
