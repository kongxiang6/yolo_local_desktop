from __future__ import annotations

import tkinter as tk


class VerticalScrolledFrame(tk.Frame):
    _active_instance: "VerticalScrolledFrame | None" = None
    _bound_root: tk.Misc | None = None

    def __init__(self, parent: tk.Widget, *, bg: str, scrollbar_width: int = 12) -> None:
        super().__init__(parent, bg=bg)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            self,
            bg=bg,
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.scrollbar = tk.Scrollbar(
            self,
            orient="vertical",
            command=self.canvas.yview,
            width=scrollbar_width,
            relief="flat",
            bd=0,
            highlightthickness=0,
        )
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.content = tk.Frame(self.canvas, bg=bg)
        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")

        self.content.bind("<Configure>", self._sync_scrollregion)
        self.canvas.bind("<Configure>", self._sync_width)
        for widget in (self, self.canvas, self.content):
            widget.bind("<Enter>", self._bind_mousewheel, add="+")
            widget.bind("<Leave>", self._unbind_mousewheel, add="+")
        self._ensure_global_mousewheel_binding()

    @classmethod
    def _resolve_from_widget(cls, widget: tk.Widget | None) -> "VerticalScrolledFrame | None":
        current = widget
        while current is not None:
            if isinstance(current, cls):
                return current
            current = getattr(current, "master", None)
        return None

    def _ensure_global_mousewheel_binding(self) -> None:
        root = self.winfo_toplevel()
        bound_root = VerticalScrolledFrame._bound_root
        if bound_root is not None:
            try:
                if bound_root.winfo_exists() and bound_root is root:
                    return
            except tk.TclError:
                pass
            VerticalScrolledFrame._bound_root = None
        root.bind_all("<MouseWheel>", VerticalScrolledFrame._dispatch_mousewheel, add="+")
        root.bind_all("<Button-4>", VerticalScrolledFrame._dispatch_mousewheel, add="+")
        root.bind_all("<Button-5>", VerticalScrolledFrame._dispatch_mousewheel, add="+")
        VerticalScrolledFrame._bound_root = root

    @classmethod
    def _dispatch_mousewheel(cls, event: tk.Event) -> str | None:
        instance = cls._resolve_from_widget(getattr(event, "widget", None))
        if instance is None:
            instance = cls._active_instance
        if instance is None:
            return None
        return instance._on_mousewheel(event)

    def _bind_mousewheel(self, _: tk.Event | None = None) -> None:
        VerticalScrolledFrame._active_instance = self

    def _unbind_mousewheel(self, _: tk.Event | None = None) -> None:
        pointer_x = self.winfo_pointerx()
        pointer_y = self.winfo_pointery()
        widget_under_pointer = self.winfo_containing(pointer_x, pointer_y)
        if widget_under_pointer and str(widget_under_pointer).startswith(str(self)):
            return
        if VerticalScrolledFrame._active_instance is self:
            VerticalScrolledFrame._active_instance = None

    def _sync_scrollregion(self, _: tk.Event) -> None:
        bbox = self.canvas.bbox("all")
        if bbox is not None:
            self.canvas.configure(scrollregion=bbox)

    def _sync_width(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> str | None:
        if VerticalScrolledFrame._active_instance not in {None, self}:
            return None
        if not self.winfo_ismapped():
            return None
        first, last = self.canvas.yview()
        if first <= 0.0 and last >= 1.0:
            return None
        delta = 0
        if getattr(event, "delta", 0):
            delta = -1 if event.delta > 0 else 1
        elif getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        if delta:
            self.canvas.yview_scroll(delta, "units")
            return "break"
        return None
