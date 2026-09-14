"""Shared visual theme for the Tkinter ML lab applications."""
import tkinter as tk
from tkinter import ttk

BG="#f4f7fb"; CARD="#ffffff"; TEXT="#172033"; MUTED="#62708a"; BORDER="#d9e1ec"; ACCENT="#4f46e5"; SUCCESS="#087f5b"

def setup_theme(root, accent=ACCENT):
    root.configure(bg=BG)
    root.option_add("*Font", ("Segoe UI",10))
    root.option_add("*tearOff", False)
    style=ttk.Style(root)
    try: style.theme_use("clam")
    except tk.TclError: pass
    style.configure(".",background=BG,foreground=TEXT,font=("Segoe UI",10))
    style.configure("TFrame",background=BG)
    style.configure("TLabel",background=BG,foreground=TEXT)
    style.configure("Title.TLabel",font=("Segoe UI Semibold",22),foreground=TEXT,background=BG)
    style.configure("Muted.TLabel",foreground=MUTED,background=BG)
    style.configure("TButton",background=CARD,foreground=TEXT,padding=(14,9),borderwidth=1,relief="flat")
    style.map("TButton",background=[("active","#e8ecff"),("pressed","#dfe4ff")])
    style.configure("Primary.TButton",background=accent,foreground="white",font=("Segoe UI Semibold",10),padding=(16,10),borderwidth=0)
    style.map("Primary.TButton",background=[("active","#3730a3"),("pressed","#312e81"),("disabled","#a5b4fc")])
    style.configure("TEntry",fieldbackground=CARD,foreground=TEXT,padding=8,bordercolor=BORDER,lightcolor=BORDER,darkcolor=BORDER)
    style.configure("TSpinbox",fieldbackground=CARD,foreground=TEXT,padding=7,bordercolor=BORDER)
    style.configure("TCombobox",fieldbackground=CARD,foreground=TEXT,padding=7)
    style.configure("TNotebook",background=BG,borderwidth=0,tabmargins=(0,6,0,0))
    style.configure("TNotebook.Tab",background="#e9eef6",foreground=MUTED,padding=(18,10),font=("Segoe UI Semibold",10),borderwidth=0)
    style.map("TNotebook.Tab",background=[("selected",CARD),("active","#eef2ff")],foreground=[("selected",accent),("active",TEXT)])
    style.configure("TLabelframe",background=CARD,bordercolor=BORDER,relief="solid",borderwidth=1,padding=12)
    style.configure("TLabelframe.Label",background=CARD,foreground=TEXT,font=("Segoe UI Semibold",11))
    style.configure("Treeview",background=CARD,fieldbackground=CARD,foreground=TEXT,rowheight=29,borderwidth=0)
    style.configure("Treeview.Heading",background="#eef2f7",foreground=TEXT,font=("Segoe UI Semibold",10),padding=8)
    style.map("Treeview",background=[("selected",accent)],foreground=[("selected","white")])
    return style

def polish(root, accent=ACCENT):
    """Polish native tk widgets after an app has constructed its interface."""
    primary=("train","evaluate","run ","classify","predict","prepare")
    def visit(widget):
        try:
            if isinstance(widget,tk.Text):
                widget.configure(bg=CARD,fg=TEXT,insertbackground=TEXT,selectbackground=accent,relief="flat",bd=0,
                                 highlightthickness=1,highlightbackground=BORDER,highlightcolor=accent,padx=14,pady=12,
                                 spacing1=2,spacing3=3)
            elif isinstance(widget,tk.Entry):
                widget.configure(bg=CARD,fg=TEXT,insertbackground=TEXT,relief="solid",bd=1,highlightthickness=1,highlightbackground=BORDER,highlightcolor=accent)
            elif isinstance(widget,tk.Button):
                label=str(widget.cget("text")).lower(); is_primary=label.startswith(primary)
                widget.configure(bg=accent if is_primary else CARD,fg="white" if is_primary else TEXT,
                                 activebackground="#3730a3" if is_primary else "#eef2ff",activeforeground="white" if is_primary else TEXT,
                                 relief="flat",bd=0,padx=14,pady=9,cursor="hand2",font=("Segoe UI Semibold",10))
            elif isinstance(widget,tk.Label):
                size=widget.cget("font"); fg=str(widget.cget("fg"))
                if fg.lower()=="white": widget.configure(bg=accent,pady=18)
            elif isinstance(widget,tk.Frame): widget.configure(bg=BG)
        except tk.TclError: pass
        for child in widget.winfo_children(): visit(child)
    visit(root)
