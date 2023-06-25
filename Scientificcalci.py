import math
import tkinter as tk

def calculate():
    expression = entry.get()
    try:
        result = eval_expression(expression)
        entry.delete(0, tk.END)
        entry.insert(tk.END, str(result))
    except:
        entry.delete(0, tk.END)
        entry.insert(tk.END, "Error")

def clear():
    entry.delete(0, tk.END)

def insert_value(value):
    entry.insert(tk.END, value)

def eval_expression(expression):
    expression = expression.replace("sqrt", "math.sqrt")
    expression = expression.replace("sin", "math.sin")
    expression = expression.replace("cos", "math.cos")
    expression = expression.replace("tan", "math.tan")
    expression = expression.replace("log", "math.log10")
    return eval(expression)

root = tk.Tk()
root.title("Scientific Calculator")

entry = tk.Entry(root, width=30)
entry.grid(row=0, column=0, columnspan=4, padx=10, pady=10)
buttons = [
    "7", "8", "9", "/",
    "4", "5", "6", "*",
    "1", "2", "3", "-",
    "0", ".", "=", "+",
    "sin", "cos", "tan", "sqrt",
    "log", "(", ")", "Clear"
]

row = 1
col = 0

for button in buttons:
    if button == "Clear":
        tk.Button(root, text=button,bg="yellow", padx=20, pady=10, command=clear).grid(row=row, column=col)
    elif button == "=":
        tk.Button(root, text=button,bg="yellow", padx=20, pady=10, command=calculate).grid(row=row, column=col)
    else:
        tk.Button(root, text=button,bg="yellow", padx=20, pady=10, command=lambda value=button: insert_value(value)).grid(row=row, column=col)

    col += 1
    if col > 3:
        col = 0
        row += 1

root.mainloop()
