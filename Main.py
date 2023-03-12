# Name: Adam Scott
# Student ID: 1423886

import tkinter as tk
import os

root = tk.Tk()
root.title("Data Viewer")
root.geometry("300x200")  # set window size


def open_historical_data():
    import MapPlot
    MapPlot.handle_selection()


def open_predictions():
    os.system("python Predict.py")


def login():
    username = username_entry.get()
    password = password_entry.get()
    if username == "a" and password == "a":
        # if the username and password match, show the main window
        login_window.destroy()
        root.deiconify()
    else:
        # if the username and password do not match, show an error message
        error_label.config(text="Invalid username or password")

'''
def open_data_docs():
    import Documentation
    data_docs_window = tk.Toplevel(root)
    data_docs_window.title("Data Documentation")

    # create buttons for different functions
    accuracy_button = tk.Button(data_docs_window, text="Accuracy", command=Documentation.run_accuracy)
    data_comparison_button = tk.Button(data_docs_window, text="Data Comparison", command=Documentation.run_data_comparison)
    prediction_scatter_button = tk.Button(data_docs_window, text="Prediction Scatter Plot", command=Documentation.run_prediction_scatter)
    actual_scatter_button = tk.Button(data_docs_window, text="Actual Scatter Plot", command=Documentation.run_actual_scatter)

    # position all the widgets in the data docs window
    accuracy_button.pack(pady=10)
    data_comparison_button.pack(pady=10)
    prediction_scatter_button.pack(pady=10)
    actual_scatter_button.pack(pady=10)
'''

# create a separate window for login
login_window = tk.Toplevel(root)
login_window.title("Login")

# add padding to the login window
login_window.geometry("250x175")
login_window.configure(padx=20, pady=20)

# create username and password entry fields
username_label = tk.Label(login_window, text="Username:")
username_entry = tk.Entry(login_window)
password_label = tk.Label(login_window, text="Password:")
password_entry = tk.Entry(login_window, show="*")

# create login and error labels
login_button = tk.Button(login_window, text="Login", command=login)
error_label = tk.Label(login_window, text="")

# position all the widgets in the login window
username_label.pack()
username_entry.pack()
password_label.pack()
password_entry.pack()
login_button.pack(pady=10)
error_label.pack()

# hide the main window until the user logs in
root.withdraw()


# add buttons to the main window
historical_data_button = tk.Button(root, text="View Historical Data", command=open_historical_data,
                                   height=3, width=20)  # set button size
historical_data_button.pack(pady=10)

predictions_button = tk.Button(root, text="View Predictions", command=open_predictions,
                               height=3, width=20)  # set button size
predictions_button.pack(pady=10)
'''
# add button for opening data docs
data_docs_button = tk.Button(root, text="Data Documentation", command=open_data_docs,
                             height=3, width=20)  # set button size
data_docs_button.pack(pady=10)
'''
root.mainloop()
