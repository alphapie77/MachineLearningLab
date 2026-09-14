"""Windowed entry point with a visible error message if startup fails."""
import traceback
try:
    import tkinter as tk
    from kmeans_app import KMeansApp
    root = tk.Tk()
    KMeansApp(root)
    root.mainloop()
except Exception:
    import ctypes
    ctypes.windll.user32.MessageBoxW(None, traceback.format_exc(), 'K-Means startup error', 16)
