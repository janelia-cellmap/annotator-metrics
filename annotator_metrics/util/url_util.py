from IPython import get_ipython
from IPython.core.display import display, HTML


def isnotebook():
    # https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def display_url(url, message):
    if isnotebook():
        display(HTML(f"""<a href="{url}">{message}.</a>"""),)
    else:
        print(f"{message}: {url}")

