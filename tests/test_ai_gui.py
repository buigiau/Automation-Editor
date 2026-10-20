import tkinter as tk
from autoedit.ai.gui import Application
from autoedit.config import load_config
from autoedit.presets import defaults


def test_preset_ui_starts_with_session_only_references(monkeypatch):
    import autoedit.ai.gui as gui
    monkeypatch.setattr(gui, 'load_settings', defaults)
    cfg = load_config(None)
    cfg['video']['characters']['reference_images'] = ['stale-reference.png']
    root = tk.Tk()
    root.withdraw()
    try:
        app = Application(root, cfg)
        root.update_idletasks()
        assert app.references == []
        assert app.preset.get() == 'Soda Pop'
        assert app.run_button['text'] == 'Chạy'
        assert not app.active
    finally:
        root.destroy()
