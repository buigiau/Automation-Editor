"""Shared desktop palette, using native Tk widgets without extra UI dependencies."""
from tkinter import ttk
import sys

BG = '#090909'
PANEL = '#141414'
FIELD = '#0d0d0d'
BORDER = '#303030'
TEXT = '#eeeeee'
MUTED = '#a3a3a3'
ACCENT = '#ffffff'
PRIMARY = '#ededed'
TINT = '#202020'
SELECTED = '#323232'
DANGER = '#eeeeee'
WARNING = '#c5c5c5'


def prepare_dpi():
    if sys.platform == 'win32':
        import ctypes
        # Call before creating Tk so fonts and window bounds use the same pixels.
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass


def dark_titlebar(root):
    if sys.platform != 'win32':
        return
    def apply(event):
        if event.widget is not root:
            return
        import ctypes
        try:
            parent = ctypes.windll.user32.GetParent
            parent.argtypes = [ctypes.c_void_p]
            parent.restype = ctypes.c_void_p
            handle = parent(root.winfo_id())
            enabled = ctypes.c_int(1)
            set_attribute = ctypes.windll.dwmapi.DwmSetWindowAttribute
            set_attribute.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
            set_attribute(handle, 20, ctypes.byref(enabled), ctypes.sizeof(enabled))
        except (AttributeError, OSError):
            pass
    root.bind('<Map>', apply, add='+')


def configure(root):
    root.configure(background=BG)
    dark_titlebar(root)
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', background=PANEL, foreground=TEXT, font=('Segoe UI', 10),
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
    style.configure('TFrame', background=PANEL)
    style.configure('Panel.TFrame', background=PANEL, borderwidth=1, relief='solid')
    style.configure('Body.TFrame', background=BG)
    style.configure('TLabel', background=PANEL, foreground=TEXT)
    style.configure('Muted.TLabel', foreground=MUTED)
    style.configure('Title.TLabel', background=BG, foreground=TEXT, font=('Segoe UI Semibold', 21))
    style.configure('Subtitle.TLabel', background=BG, foreground=MUTED)
    style.configure('Section.TLabel', font=('Segoe UI Semibold', 11))
    style.configure('Metric.TLabel', font=('Consolas', 19), foreground=ACCENT)
    style.configure('Badge.TLabel', background=TINT, foreground=TEXT, font=('Consolas', 10), padding=(12, 7))
    style.configure('TButton', background=TINT, foreground=TEXT, font=('Segoe UI', 9),
                    padding=(12, 8), borderwidth=1, relief='flat')
    style.map('TButton', background=[('disabled', '#181818'), ('active', '#353535')],
              foreground=[('disabled', '#777777')], bordercolor=[('focus', '#898989')])
    style.configure('Primary.TButton', background=PRIMARY, foreground='#111111', font=('Segoe UI Semibold', 10),
                    bordercolor=PRIMARY, lightcolor=PRIMARY, darkcolor=PRIMARY)
    style.map('Primary.TButton', background=[('disabled', '#353535'), ('active', '#d0d0d0')],
              foreground=[('disabled', '#929292')])
    style.configure('Danger.TButton', foreground=DANGER, background=FIELD)
    style.map('Danger.TButton', background=[('disabled', '#181818'), ('active', '#393939')])
    style.configure('Active.TButton', background='#383838', foreground=TEXT)
    style.configure('TEntry', fieldbackground=FIELD, foreground=TEXT, insertcolor=TEXT, padding=6,
                    lightcolor=BORDER, darkcolor=BORDER)
    style.map('TEntry', fieldbackground=[('readonly', FIELD)])
    style.configure('TCombobox', fieldbackground=FIELD, foreground=TEXT, arrowcolor=MUTED, padding=7,
                    background=TINT)
    style.map('TCombobox', fieldbackground=[('readonly', FIELD)], foreground=[('readonly', TEXT)],
              selectbackground=[('readonly', FIELD)], selectforeground=[('readonly', TEXT)])
    root.option_add('*TCombobox*Listbox.background', FIELD)
    root.option_add('*TCombobox*Listbox.foreground', TEXT)
    root.option_add('*TCombobox*Listbox.selectBackground', SELECTED)
    root.option_add('*TCombobox*Listbox.selectForeground', TEXT)
    style.configure('Treeview', background=FIELD, fieldbackground=FIELD, foreground=TEXT,
                    rowheight=36, borderwidth=0)
    style.map('Treeview', background=[('selected', SELECTED)], foreground=[('selected', '#ffffff')])
    style.configure('Treeview.Heading', background=TINT, foreground=MUTED,
                    font=('Segoe UI Semibold', 9), padding=(8, 10), relief='flat')
    style.map('Treeview.Heading', background=[('active', '#333333')])
    style.configure('TProgressbar', background=ACCENT, troughcolor='#242424', borderwidth=0, thickness=3)
    style.configure('TNotebook', background=BG, borderwidth=0)
    style.configure('TNotebook.Tab', padding=(16, 10))
    style.map('TNotebook.Tab', background=[('selected', PANEL)], foreground=[('selected', PRIMARY)])
    style.configure('TLabelframe', background=PANEL, bordercolor=BORDER)
    style.configure('TLabelframe.Label', background=PANEL, foreground=MUTED)
    style.configure('TScrollbar', background='#3a3a3a', troughcolor=FIELD, arrowcolor=MUTED,
                    borderwidth=0, arrowsize=10)
    style.map('TScrollbar', background=[('active', '#606060')])
    style.configure('TCheckbutton', background=PANEL, foreground=TEXT, indicatorbackground=FIELD)
    style.map('TCheckbutton', background=[('active', PANEL)], indicatorbackground=[('selected', '#dddddd')])


def text_options():
    return dict(background=FIELD, foreground=TEXT, insertbackground=TEXT,
                selectbackground=SELECTED, selectforeground=TEXT, relief='flat',
                highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT,
                padx=12, pady=10)


def list_options():
    options = text_options()
    for key in ('insertbackground', 'padx', 'pady'):
        options.pop(key)
    return options
