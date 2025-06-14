import tkinter as tk
from util.vertex import Vertex
from util.segment import Segment
from util.sketch import Sketch
import numpy as np


class OrigamiGUI:
    sketch: Sketch
    process_adding_vertex: bool

    def __init__(self, root):
        self.sketch = Sketch()
        self.root = root
        self.root.title("Tkinter Canvas Demo")

        # Frame principale
        main_frame = tk.Frame(root)
        main_frame.pack(padx=10, pady=10)

        # Frame laterale per i bottoni
        button_frame = tk.Frame(main_frame)
        button_frame.pack(side=tk.LEFT, padx=(0, 10))

        # Bottoni verticali con larghezza fissa
        self.btn_clear = tk.Button(
            button_frame, text="Pulisci canvas", width=20, command=self.clear_canvas
        )
        self.btn_clear.pack(pady=5)

        self.btn_info = tk.Button(
            button_frame, text="Info", width=20, command=self.show_info
        )
        self.btn_info.pack(pady=5)

        self.btn_exit = tk.Button(
            button_frame, text="Esci", width=20, command=self.root.quit
        )
        self.btn_exit.pack(pady=5)

        self.btn_exit = tk.Button(
            button_frame, text="Add Vertex", width=20, command=self.action_add_vertex
        )
        self.btn_exit.pack(pady=5)

        # Canvas a destra
        self.canvas = tk.Canvas(main_frame, width=500, height=500, bg="white")
        self.canvas.pack(side=tk.RIGHT)

        # Binding eventi mouse
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)

    def on_click(self, event):
        x, y = event.x, event.y
        if self.process_adding_vertex:
            
        print(f"Click a ({x}, {y})")
        self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="blue")

    def on_drag(self, event):
        x, y = event.x, event.y
        print(f"Trascinamento a ({x}, {y})")
        self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="red")

    def clear_canvas(self):
        self.canvas.delete("all")

    def action_add_vertex(self):
        self.process_adding_vertex = not self.process_adding_vertex

    def show_info(self):
        info_window = tk.Toplevel(self.root)

        info_window.title("Informazioni")
        info_window.geometry("300x150")  # larghezza x altezza

        label = tk.Label(
            info_window,
            text="Questa è una demo con canvas e bottoni.\nScritta in Python con Tkinter.",
            padx=10,
            pady=10,
        )
        label.pack()

        close_btn = tk.Button(info_window, text="Chiudi", command=info_window.destroy)
        close_btn.pack(pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = OrigamiGUI(root)
    root.mainloop()
