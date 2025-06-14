import tkinter as tk
from util.vertex import Vertex
from util.segment import Segment
from util.sketch import Sketch
from util.constraints.fixed import Fixed
import numpy as np


class OrigamiGUI:
    sketch: Sketch
    process_adding_vertex: bool
    process_adding_segment: bool
    process_adding_segment_vertex_candidate: Vertex
    process_adding_segment_vertex_count: int
    process_fixing_vertex: bool

    def __init__(self, root):
        self.process_adding_vertex = False
        self.process_adding_segment = False
        self.process_fixing_vertex = False
        self.sketch = Sketch()
        self.root = root
        self.process_adding_segment_vertex_count = 0

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

        self.btn_add_vertex = tk.Button(
            button_frame, text="Add Vertex", width=20, command=self.action_add_vertex
        )
        self.btn_add_vertex.pack(pady=5)

        self.btn_add_segment = tk.Button(
            button_frame, text="Add Segment", width=20, command=self.action_add_segment
        )
        self.btn_add_segment.pack(pady=5)

        self.btn_fix_vertex = tk.Button(
            button_frame, text="Fix Vertex", width=20, command=self.action_fix_vertex
        )
        self.btn_fix_vertex.pack(pady=5)

        # Canvas a destra
        self.canvas = tk.Canvas(main_frame, width=500, height=500, bg="white")
        self.canvas.pack(side=tk.RIGHT)

        # Binding eventi mouse
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)

    def on_click(self, event):
        x, y = event.x, event.y
        if self.process_adding_vertex:
            self.sketch.add_vertex(Vertex(np.array([x, y])))
        elif self.process_adding_segment:
            v = self.sketch.get_closer_vertex(np.array([x, y]))
            if v is not None:
                if self.process_adding_segment_vertex_count == 0:
                    self.process_adding_segment_vertex_count = 1
                    self.process_adding_segment_vertex_candidate = v
                elif self.process_adding_segment_vertex_count == 1:
                    self.process_adding_segment_vertex_count = 0
                    s = Segment(self.process_adding_segment_vertex_candidate, v)
                    self.sketch.add_segment(s)
        elif self.process_fixing_vertex:
            v = self.sketch.get_closer_vertex(np.array([x, y]))
            if v is not None:
                if not self.sketch.is_fixed(v):
                    self.sketch.add_constraint(Fixed(v, v.position))
        self.redraw_graphics()

    def on_drag(self, event):
        newpos = np.array([event.x, event.y])
        v = self.sketch.get_closer_vertex(newpos)
        if v is not None:
            if not self.sketch.is_fixed(v):
                v.position = newpos
        self.redraw_graphics()

    def redraw_graphics(self):
        self.canvas.delete("all")
        for s in self.sketch.segments:
            x1, y1 = s.v1.position[0], s.v1.position[1]
            x2, y2 = s.v2.position[0], s.v2.position[1]
            self.canvas.create_line(x1, y1, x2, y2, fill="blue")
        for v in self.sketch.vertices:
            x, y = v.position[0], v.position[1]
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="red")
        for f in self.sketch.get_fixed_constraints():
            x, y = f.v.position[0], f.v.position[1]
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="black")
        self.btn_add_vertex.config(
            text="Quit" if self.process_adding_vertex else "Add Vertex",
            # highlightbackground="red" if self.process_adding_vertex else "grey",
        )
        self.btn_add_segment.config(
            text="Quit" if self.process_adding_segment else "Add Segment",
            # highlightbackground="red" if self.process_adding_vertex else "grey",
        )
        self.btn_fix_vertex.config(
            text="Quit" if self.process_fixing_vertex else "Fix Vertex",
            # highlightbackground="red" if self.process_adding_vertex else "grey",
        )

    def clear_canvas(self):
        self.canvas.delete("all")

    def action_add_vertex(self):
        self.process_adding_vertex = not self.process_adding_vertex
        self.process_adding_segment = False
        self.process_fixing_vertex = False
        self.redraw_graphics()

    def action_add_segment(self):
        self.process_adding_segment = not self.process_adding_segment
        self.process_adding_segment_vertex_count = 0
        self.process_adding_vertex = False
        self.process_fixing_vertex = False
        self.redraw_graphics()

    def action_fix_vertex(self):
        self.process_fixing_vertex = not self.process_fixing_vertex
        self.process_adding_segment_vertex_count = 0
        self.process_adding_vertex = False
        self.process_adding_segment = False
        self.redraw_graphics()

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
