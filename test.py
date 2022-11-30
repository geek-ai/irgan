# importing libraries
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import time
import matplotlib.pyplot as plt

# creating initial data values
# of x and y
x = np.array([1])
y = np.array([1])

# to run GUI event loop
plt.ion()

# here we are creating sub plots
figure, ax = plt.subplots(figsize=(10, 8))
line1, = ax.plot(x, y)

# setting title
plt.title("Geeks For Geeks", fontsize=20)

# setting x-axis label and y-axis label
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# Loop
for i in range(50):
    # creating new Y values
    x = np.append(x, i)
    y = np.append(y, i)

    # updating data values
    line1.set_xdata(x)
    line1.set_ydata(y)

    # drawing updated values
    figure.canvas.draw()

    # This will run the GUI event
    # loop until all UI events
    # currently waiting have been processed
    figure.canvas.flush_events()

    time.sleep(0.1)
    plt.show()