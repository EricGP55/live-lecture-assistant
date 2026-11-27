#!/usr/bin/env python3
"""Small always-on-top native window showing the /mini page (pywebview, WebKit).

The window has no title bar: its first line is the page's own bar, which can be dragged
to move the window, double-clicked (or ▾) to collapse it to that single line, and ✕ closes it.

  python mini_window.py http://127.0.0.1:8765/mini
"""
import sys

import webview

WIDTH, HEIGHT, BAR = 380, 560, 34


class Api:
    """Called from the page's JavaScript as window.pywebview.api.*"""

    def __init__(self):
        self.window = None
        self.expanded_height = HEIGHT

    def set_collapsed(self, collapsed):
        w = self.window
        if collapsed:
            if w.height > BAR + 20:
                self.expanded_height = w.height   # remember the size to restore it
            w.resize(w.width, BAR)
        else:
            w.resize(w.width, max(self.expanded_height, 200))

    def close(self):
        self.window.destroy()


url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/mini"
api = Api()
api.window = webview.create_window(
    "Class · mini", url, js_api=api, width=WIDTH, height=HEIGHT, min_size=(260, BAR),
    on_top=True, frameless=True, easy_drag=False, background_color="#0b0d11", text_select=True)
webview.start()
