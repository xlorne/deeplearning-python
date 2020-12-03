# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import mxnet as mx
from mxnet import nd

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    data = [nd.ones((1, 2), ctx=mx.gpu(i)) * (i + 1) for i in range(2)]
    print(data)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
