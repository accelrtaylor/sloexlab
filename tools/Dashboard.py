import ipywidgets as widgets

from tools.TheoryDashboard import *
from tools.TrackDashboard import *
from tools.VisualiserDashboard import *
from tools.CreditFooter import *

def Dashboard():
    'Takes results from TheoryDashboard, TrackDashboard and Visualiser in three tabs'
    Tab = widgets.Tab()
    
    Tab.children = [TheoryDashboard(), TrackDashboard(), Visualiser()]
    
    Tab.set_title(0, 'Theoretical')
    Tab.set_title(1, 'Tracking Code')
    Tab.set_title(2, 'Visualiser')
    
    Dashboard = widgets.VBox([Tab, CreditDashboard()])
    display(Dashboard)
