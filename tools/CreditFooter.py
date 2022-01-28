import ipywidgets as widgets

def CreditDashboard():
    'Acknowledgements to contributers'
    Credits = widgets.Button(description='Credits', icon='address-card')
    Credits_output = widgets.Output(layout={'border': '1px solid black'})
    credit = widgets.HBox([Credits, Credits_output])

    def credit_push(b):
        with Credits_output:
            if Credits_output.outputs == ():
                Credits_output.clear_output()
                display(widgets.HTML('<p> Sloex lab made by: Rebecca Taylor (<a href="rebecca.taylor@cern.ch">rebecca.taylor@cern.ch</a>) </p> With contributions from: Pablo Arrutia (<a href="pablo.andreas.arrutia.sota@cern.ch">pablo.andreas.arrutia.sota@cern.ch</a>)'))
            else:
                Credits_output.clear_output()
    Credits.on_click(credit_push)
    return(credit)