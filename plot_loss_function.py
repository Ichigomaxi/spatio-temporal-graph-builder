import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

import numpy as np
import pandas as pd

loss_path = r"C:\Users\maxil\Downloads\run-.-tag-loss.csv"
loss_val_path = r"C:\Users\maxil\Downloads\run-.-tag-loss_val.csv"

df = pd.read_csv(loss_path)


plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
columns = ["Wall time", "Step",  "Value"]
# df = pd.read_csv("input.csv", usecols=columns)
print("Contents in csv file:\n", df)
# plt.plot(df.Step, df.Value)
# plt.show()

df_vall = pd.read_csv(loss_val_path)
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
print("Contents in csv file:\n", df_vall)
# plt.plot(df_vall.Step, df_vall.Value)
# plt.show()


plt.plot(df.Step, df.Value, 'b', label='Training Loss')

plt.plot(df_vall.Step, df_vall.Value, color='orange', linewidth=3,label='Validation Loss')

plt.title('Training and Validation loss')

plt.xlabel('Steps')

plt.ylabel('Loss')

plt.legend()

plt.show()