import tkinter as tk
import time
def modf_time():
    c_time = time.strftime("%H:%M:%S %p")
    time_Label.config(text=c_time)
    window.after(1000, modf_time)
window = tk.Pk()
window.title("Digital Clock")
window.configure(bg="black")
time_Label = tk.Label(window, font=("Arial",40), fg="white",bg="black")
time_Label.pack(pady=50)
modf_time()
window.mainloop()