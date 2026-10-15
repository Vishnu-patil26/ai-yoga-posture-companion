"""The Tk toolkit behind every desktop screen (launcher.py and collect.py).

One black theme, built once here, so a screen is only layout:

* ``setup(root)`` makes the window sharp on a scaled display (DPI awareness),
  picks the fonts, and styles every ttk widget.
* ``px()`` turns a design pixel into a real pixel, so spacing and image sizes
  keep their proportions on a 125% or 150% display.
* plain helpers - ``label``, ``button``, ``entry``, ``pill``, ``card``, ``stat``,
  ``alert``, ``Hairline``, ``Toggle``, ``Tabs``, ``ScrollArea`` - each returns an
  ordinary Tk widget.

Containers are ``tk.Frame`` (not ``ttk.Frame``) so a child can always ask its
parent for the background it should sit on.
"""

from __future__ import annotations

import ctypes
import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

from . import theme
from .theme import TOKENS as T

SCALE = 1.0
F: dict[str, tkfont.Font] = {}          #: named fonts, filled by setup()
_SCROLLERS: dict[str, "ScrollArea"] = {}


def init_dpi() -> None:
    """Ask Windows not to bitmap-stretch this process.  Call before the first Tk()."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def dark_titlebar(window: tk.Misc) -> None:
    """Make the Windows title bar black to match the page (Windows 10 1809+ / 11).

    Harmless anywhere else: on other systems, or if the call is refused, the
    window simply keeps its normal title bar.
    """
    if sys.platform != "win32":
        return
    try:
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        dwm = ctypes.windll.dwmapi.DwmSetWindowAttribute
        on = ctypes.c_int(1)
        for attr in (20, 19):                                 # immersive dark mode (new, old id)
            if dwm(hwnd, attr, ctypes.byref(on), ctypes.sizeof(on)) == 0:
                break
        caption = ctypes.c_int(0x00050505)                    # Windows 11: caption and text colours
        text = ctypes.c_int(0x00F2F2F2)
        dwm(hwnd, 35, ctypes.byref(caption), ctypes.sizeof(caption))
        dwm(hwnd, 36, ctypes.byref(text), ctypes.sizeof(text))
    except Exception:
        pass


def px(n: float) -> int:
    """A design pixel (96 dpi) as a real pixel on this display."""
    return max(1, int(round(n * SCALE)))


def _pick(root: tk.Misc, wanted: tuple[str, ...], fallback: str) -> str:
    have = {f.lower(): f for f in tkfont.families(root)}
    for name in wanted:
        if name.lower() in have:
            return have[name.lower()]
    return fallback


def setup(root: tk.Tk) -> None:
    """Scale, fonts and ttk styles for the black theme.  Idempotent per root."""
    global SCALE
    SCALE = max(1.0, root.winfo_fpixels("1i") / 96.0)
    root.configure(bg=T["bg"])

    sans = _pick(root, theme.SANS_FAMILIES, "TkDefaultFont")
    mono = _pick(root, theme.MONO_FAMILIES, "TkFixedFont")
    mk = lambda fam, size, weight="normal": tkfont.Font(      # noqa: E731
        root=root, family=fam, size=size, weight=weight)
    F.clear()
    F.update(
        body=mk(sans, 10), small=mk(sans, 9), bold=mk(sans, 10, "bold"),
        h1=mk(sans, 22, "bold"), h2=mk(sans, 13, "bold"), h3=mk(sans, 11, "bold"),
        mono=mk(mono, 9), mono_b=mk(mono, 9, "bold"), cap=mk(mono, 8), cap_b=mk(mono, 8, "bold"),
        brand=mk(mono, 10, "bold"), big=mk(mono, 54, "bold"), num=mk(sans, 24),
        stat=mk(sans, 22),
    )
    root.option_add("*Font", F["body"])
    root.option_add("*TCombobox*Listbox.background", T["card2"])
    root.option_add("*TCombobox*Listbox.foreground", T["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", T["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", T["bg"])
    root.option_add("*TCombobox*Listbox.font", F["body"])
    root.option_add("*TCombobox*Listbox.borderWidth", 0)
    root.option_add("*Menu.background", T["card2"])
    root.option_add("*Menu.foreground", T["text"])

    st = ttk.Style(root)
    st.theme_use("clam")
    flat = dict(lightcolor=T["bg"], darkcolor=T["bg"], bordercolor=T["border"], relief="flat")
    st.configure(".", background=T["bg"], foreground=T["text"], font=F["body"],
                 troughcolor=T["card"], focuscolor=T["bg"], selectbackground=T["accent"],
                 selectforeground=T["bg"], **flat)

    # ---- buttons: monospaced, square; primary = white on black
    pad = (px(16), px(8))
    st.configure("TButton", background=T["card2"], foreground=T["text"], font=F["mono"],
                 padding=pad, bordercolor=T["border"], lightcolor=T["card2"],
                 darkcolor=T["card2"], focuscolor=T["card2"], anchor="center")
    st.map("TButton", background=[("disabled", T["card"]), ("pressed", T["border"]), ("active", "#1c1c1c")],
           foreground=[("disabled", T["dim"])],
           bordercolor=[("disabled", T["border"]), ("focus", T["muted"]), ("active", T["muted"])])
    st.configure("Primary.TButton", background=T["accent"], foreground=T["bg"], font=F["mono_b"],
                 padding=pad, bordercolor=T["accent"], lightcolor=T["accent"],
                 darkcolor=T["accent"], focuscolor=T["accent"])
    st.map("Primary.TButton", background=[("disabled", T["border"]), ("pressed", "#bdbdbd"), ("active", "#dcdcdc")],
           foreground=[("disabled", T["dim"])],
           bordercolor=[("disabled", T["border"]), ("active", "#dcdcdc")])
    st.configure("Ghost.TButton", background=T["bg"], foreground=T["muted"], font=F["mono"],
                 padding=pad, bordercolor=T["border"], lightcolor=T["bg"], darkcolor=T["bg"],
                 focuscolor=T["bg"])
    st.map("Ghost.TButton", background=[("active", T["card"])],
           foreground=[("disabled", T["dim"]), ("active", T["text"])],
           bordercolor=[("active", T["muted"]), ("focus", T["muted"])])
    st.configure("Side.TButton", background=T["card"], foreground=T["muted"], font=F["mono"],
                 padding=(px(10), px(6)), bordercolor=T["border"], lightcolor=T["card"],
                 darkcolor=T["card"], focuscolor=T["card"], anchor="w")
    st.map("Side.TButton", background=[("active", T["card2"])],
           foreground=[("active", T["text"])], bordercolor=[("active", T["muted"])])

    # ---- the plain ttk widgets, so a screen built from them (collect.py) is black too
    st.configure("TFrame", background=T["bg"])
    st.configure("TLabel", background=T["bg"], foreground=T["text"])
    st.configure("Title.TLabel", font=F["h1"])
    st.configure("Sub.TLabel", foreground=T["muted"], font=F["mono"])
    st.configure("Big.TLabel", font=F["big"], foreground=T["text"])
    st.configure("Accent.TButton", background=T["accent"], foreground=T["bg"], font=F["mono_b"],
                 padding=pad, bordercolor=T["accent"], lightcolor=T["accent"],
                 darkcolor=T["accent"], focuscolor=T["accent"])
    st.map("Accent.TButton", background=[("disabled", T["border"]), ("pressed", "#bdbdbd"), ("active", "#dcdcdc")],
           foreground=[("disabled", T["dim"])], bordercolor=[("disabled", T["border"]), ("active", "#dcdcdc")])
    for name in ("TLabelframe", "Card.TLabelframe"):
        st.configure(name, background=T["bg"], bordercolor=T["border"], lightcolor=T["border"],
                     darkcolor=T["border"], relief="solid", borderwidth=1)
        st.configure(name + ".Label", background=T["bg"], foreground=T["muted"], font=F["cap_b"])
    for name in ("TCheckbutton", "TRadiobutton"):
        st.configure(name, background=T["bg"], foreground=T["text"], indicatorbackground=T["card2"],
                     indicatorforeground=T["accent"], upperbordercolor=T["muted"],
                     lowerbordercolor=T["muted"], indicatormargin=(0, 0, px(8), 0), focuscolor=T["bg"])
        st.map(name, background=[("active", T["bg"])], foreground=[("disabled", T["dim"])],
               indicatorbackground=[("disabled", T["card"]), ("pressed", T["card2"]),
                                    ("selected", T["card2"])],
               upperbordercolor=[("selected", T["accent"]), ("focus", T["accent"])],
               lowerbordercolor=[("selected", T["accent"]), ("focus", T["accent"])])
    st.configure("TEntry", fieldbackground=T["card2"], foreground=T["text"], insertcolor=T["accent"],
                 bordercolor=T["border"], lightcolor=T["card2"], darkcolor=T["card2"],
                 padding=(px(8), px(5)))
    st.map("TEntry", bordercolor=[("focus", T["accent"])], lightcolor=[("focus", T["accent"])],
           darkcolor=[("focus", T["accent"])], fieldbackground=[("disabled", T["card"])],
           foreground=[("disabled", T["dim"])])
    st.configure("TSeparator", background=T["border"])

    # ---- inputs
    st.configure("TCombobox", fieldbackground=T["card2"], background=T["card2"], foreground=T["text"],
                 arrowcolor=T["muted"], bordercolor=T["border"], lightcolor=T["card2"],
                 darkcolor=T["card2"], selectbackground=T["card2"], selectforeground=T["text"],
                 padding=(px(8), px(5)), arrowsize=px(14))
    st.map("TCombobox", fieldbackground=[("readonly", T["card2"]), ("disabled", T["card"])],
           foreground=[("readonly", T["text"]), ("disabled", T["dim"])],
           bordercolor=[("focus", T["accent"]), ("active", T["muted"])],
           arrowcolor=[("active", T["text"])],
           selectbackground=[("readonly", T["card2"])], selectforeground=[("readonly", T["text"])])

    # ---- scrollbar: just a thumb
    st.layout("Vertical.TScrollbar", [("Vertical.Scrollbar.trough", {
        "sticky": "ns", "children": [("Vertical.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})]})])
    st.configure("Vertical.TScrollbar", background=T["border"], troughcolor=T["bg"],
                 bordercolor=T["bg"], lightcolor=T["border"], darkcolor=T["border"],
                 width=px(8), arrowsize=px(8))
    st.map("Vertical.TScrollbar", background=[("active", T["border2"]), ("pressed", T["muted"])])

    # ---- progress: thin white bar
    st.configure("Thin.Horizontal.TProgressbar", background=T["accent"], troughcolor=T["border"],
                 bordercolor=T["border"], lightcolor=T["accent"], darkcolor=T["accent"],
                 thickness=px(4))

    # ---- tree (the pose hierarchy)
    st.configure("Treeview", background=T["bg"], fieldbackground=T["bg"], foreground=T["text"],
                 bordercolor=T["border"], rowheight=px(26), font=F["body"], borderwidth=0)
    st.map("Treeview", background=[("selected", T["card2"])], foreground=[("selected", T["text"])])
    st.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
    st.configure("Treeview.Heading", background=T["card"], foreground=T["muted"], font=F["cap"],
                 bordercolor=T["border"])

    # one wheel handler for every ScrollArea (a per-widget bind_all leaks and
    # raises TclError once that widget has been destroyed)
    root.bind_all("<MouseWheel>", _on_wheel, add="+")
    dark_titlebar(root)


# ----------------------------------------------------------------- scrolling
_NO_WHEEL = {"Text", "Treeview", "Listbox", "TCombobox"}


def _on_wheel(event: tk.Event) -> None:
    w = event.widget
    try:
        if isinstance(w, str):
            return
        if w.winfo_class() in _NO_WHEEL and str(w) not in _SCROLLERS:
            return
        while w is not None:
            area = _SCROLLERS.get(str(w))
            if area is not None:
                area.scroll(-1 if event.delta > 0 else 1, max(1, abs(event.delta) // 120))
                return
            w = w.master
    except (tk.TclError, KeyError, AttributeError):
        pass


class ScrollArea(tk.Frame):
    """A vertically scrolling region.  Put content in ``.body``."""

    def __init__(self, parent: tk.Misc, bg: str = "bg") -> None:
        super().__init__(parent, bg=T[bg])
        self._bg = T[bg]
        self.canvas = tk.Canvas(self, bg=self._bg, highlightthickness=0, bd=0)
        self.bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.body = tk.Frame(self.canvas, bg=self._bg)
        self._win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.canvas.configure(yscrollcommand=self._set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.body.bind("<Configure>", lambda _e: self._sync())
        self.canvas.bind("<Configure>", self._fit)
        _SCROLLERS[str(self.canvas)] = self
        self.bind("<Destroy>", lambda _e: _SCROLLERS.pop(str(self.canvas), None))
        self._shown = False

    def _set(self, lo, hi) -> None:
        self.bar.set(lo, hi)
        need = float(lo) > 0.0 or float(hi) < 1.0
        if need and not self._shown:
            self.bar.pack(side="right", fill="y")
            self._shown = True
        elif not need and self._shown:
            self.bar.pack_forget()
            self._shown = False

    def _fit(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._win, width=event.width)
        self._sync()

    def _sync(self) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        if self.canvas.winfo_height() >= self.body.winfo_reqheight():
            self.canvas.yview_moveto(0)

    def scroll(self, direction: int, amount: int = 1) -> None:
        if self.body.winfo_reqheight() > self.canvas.winfo_height():
            self.canvas.yview_scroll(direction * amount * 2, "units")

    def to_top(self) -> None:
        self.canvas.yview_moveto(0)


# -------------------------------------------------------------------- basics
def _bg_of(parent: tk.Misc, default: str = "bg") -> str:
    try:
        return parent.cget("bg")
    except (tk.TclError, AttributeError):
        return T[default]


def clear(box: tk.Misc) -> None:
    """Destroy everything inside ``box`` and let it collapse.

    Tk never shrinks a frame that has lost its last child - it keeps the size it
    had - so a dismissed message would leave a gap that every later screen
    inherited.  Re-requesting a 1 px size fixes that.
    """
    for w in box.winfo_children():
        w.destroy()
    try:
        box.configure(height=1, width=1)
    except tk.TclError:
        pass


def frame(parent: tk.Misc, bg: str | None = None, **kw) -> tk.Frame:
    return tk.Frame(parent, bg=T[bg] if bg else _bg_of(parent), **kw)


_KINDS = {
    "body": ("body", "text"), "bold": ("bold", "text"), "muted": ("body", "muted"),
    "small": ("small", "muted"), "mono": ("mono", "muted"), "mono_text": ("mono", "text"),
    "cap": ("cap", "muted"), "cap_text": ("cap_b", "text"), "h1": ("h1", "text"),
    "h2": ("h2", "text"), "h3": ("h3", "text"), "brand": ("brand", "text"),
    "big": ("big", "text"), "num": ("num", "text"), "stat": ("stat", "text"),
    "good": ("body", "good"), "warn": ("body", "warn"), "bad": ("body", "bad"),
}


def label(parent: tk.Misc, text: str = "", kind: str = "body", fg: str | None = None,
          bg: str | None = None, wrap: int = 0, justify: str = "left", fit: bool = False,
          **kw) -> tk.Label:
    """A text label.  ``wrap`` breaks lines at that many design pixels.  ``fit=True``
    instead wraps at however wide the label ends up - pack it with ``fill="x"`` and
    it follows the window as it is resized."""
    font, colour = _KINDS[kind]
    if kind.startswith("cap"):
        text = text.upper()
    lbl = tk.Label(parent, text=text, font=F[font], fg=T.get(fg, fg) if fg else T[colour],
                   bg=T[bg] if bg else _bg_of(parent), anchor="w", justify=justify,
                   wraplength=px(260) if fit else (px(wrap) if wrap else 0), bd=0, **kw)
    if fit:
        def _follow(e: tk.Event, lbl=lbl) -> None:
            w = max(px(80), e.width)
            if int(lbl.cget("wraplength")) != w:
                lbl.configure(wraplength=w)
        lbl.bind("<Configure>", _follow)
    return lbl


def button(parent: tk.Misc, text: str, command, kind: str = "default", **kw) -> ttk.Button:
    style = {"primary": "Primary.TButton", "ghost": "Ghost.TButton",
             "side": "Side.TButton"}.get(kind, "TButton")
    return ttk.Button(parent, text=text, command=command, style=style, takefocus=True, **kw)


def entry(parent: tk.Misc, var: tk.Variable, width: int = 24, **kw) -> tk.Entry:
    e = tk.Entry(parent, textvariable=var, width=width, font=F["body"], fg=T["text"],
                 bg=T["card2"], insertbackground=T["accent"], relief="flat", bd=0,
                 highlightthickness=1, highlightbackground=T["border"],
                 highlightcolor=T["accent"], selectbackground=T["accent"],
                 selectforeground=T["bg"], disabledbackground=T["card"],
                 disabledforeground=T["dim"], **kw)
    return e


def mark(widget: tk.Misc, bad: bool) -> None:
    """Outline an entry in red (bad) or back to its normal hairline."""
    try:
        widget.configure(highlightbackground=T["bad"] if bad else T["border"])
    except tk.TclError:
        pass


def combo(parent: tk.Misc, var: tk.Variable, values, width: int = 22) -> ttk.Combobox:
    cb = ttk.Combobox(parent, textvariable=var, values=list(values), state="readonly",
                      width=width, font=F["body"])
    cb.bind("<<ComboboxSelected>>", lambda _e: cb.selection_clear(), add="+")
    return cb


def textbox(parent: tk.Misc, height: int = 8, **kw) -> tk.Text:
    return tk.Text(parent, height=height, wrap="word", font=F["body"], fg=T["text"],
                   bg=T["card"], insertbackground=T["accent"], relief="flat", bd=0,
                   highlightthickness=1, highlightbackground=T["border"],
                   highlightcolor=T["border"], padx=px(14), pady=px(12),
                   selectbackground=T["border2"], spacing3=px(2), **kw)


_PILL = {"muted": ("muted", "border"), "text": ("text", "muted"), "good": ("good", None),
         "warn": ("warn", None), "bad": ("bad", None), "accent": ("accent", None)}


def pill(parent: tk.Misc, text: str, kind: str = "muted", bg: str | None = None) -> tk.Label:
    fg, edge = _PILL[kind]
    col = T[fg]
    return tk.Label(parent, text=text.upper(), font=F["cap_b"], fg=col,
                    bg=T[bg] if bg else _bg_of(parent), padx=px(7), pady=px(2),
                    highlightthickness=1,
                    highlightbackground=T[edge] if edge else theme.mix(col, T["bg"], 0.6), bd=0)


class Hairline(tk.Canvas):
    """The blue-orange-pink rule under a title."""

    def __init__(self, parent: tk.Misc, width: int = 56, height: int = 2) -> None:
        w, h = px(width), px(height)
        super().__init__(parent, width=w, height=h, bg=_bg_of(parent), highlightthickness=0, bd=0)
        for x in range(w):
            self.create_line(x, 0, x, h, fill=theme.glow_at(x / max(1, w - 1)))


def rule(parent: tk.Misc, color: str = "border") -> tk.Frame:
    """A one-pixel horizontal divider."""
    return tk.Frame(parent, bg=T[color], height=1)


class Card(tk.Frame):
    """A bordered box.  Put content in ``.body`` (it already has padding)."""

    def __init__(self, parent: tk.Misc, bg: str = "bg", border: str = "border",
                 pad: int = 14, **kw) -> None:
        super().__init__(parent, bg=T[bg], highlightthickness=1, highlightbackground=T[border],
                         highlightcolor=T[border], bd=0, **kw)
        self._bg = T[bg]
        self.body = tk.Frame(self, bg=T[bg])
        self.body.pack(fill="both", expand=True, padx=px(pad), pady=px(pad))

    def outline(self, colour: str) -> None:
        self.configure(highlightbackground=T.get(colour, colour), highlightcolor=T.get(colour, colour))


def stat(parent: tk.Misc, title: str, value: str, hint: str = "") -> Card:
    """A KPI tile: small capitals, a big number, an optional one-line note."""
    c = Card(parent, "bg")
    label(c.body, title, "cap").pack(anchor="w")
    label(c.body, value, "stat").pack(anchor="w", pady=(px(6), 0))
    if hint:
        label(c.body, hint, "small", wrap=200).pack(anchor="w", pady=(px(2), 0))
    return c


def stat_row(parent: tk.Misc, items) -> tk.Frame:
    """A row of equal-width KPI tiles from ``(title, value, hint)`` triples."""
    row = frame(parent)
    n = len(items)
    for i, (title, value, hint) in enumerate(items):
        row.columnconfigure(i, weight=1, uniform="stat")
        stat(row, title, value, hint).grid(row=0, column=i, sticky="nsew",
                                           padx=(0 if i == 0 else px(6), 0 if i == n - 1 else px(6)))
    return row


def alert(parent: tk.Misc, text: str, kind: str = "warn") -> Card:
    """A bordered message, coloured by how serious it is."""
    c = Card(parent, "bg", border="border", pad=10)
    c.outline(theme.mix(T.get(kind, kind), T["bg"], 0.55))
    label(c.body, text, "body", fg=kind if kind in T else None, wrap=720).pack(anchor="w")
    return c


# ---------------------------------------------------------------- the toggle
class Toggle(tk.Frame):
    """A check box or a radio dot with its caption.  Square and round, thin white
    when on.  Works with the keyboard (Tab to focus, Space or Enter to flip)."""

    SIZE = 15

    def __init__(self, parent: tk.Misc, text: str, variable: tk.Variable, value=None,
                 kind: str = "check", command=None, bg: str | None = None,
                 font: str = "body") -> None:
        super().__init__(parent, bg=T[bg] if bg else _bg_of(parent), highlightthickness=1,
                         highlightbackground=_bg_of(parent), highlightcolor=T["muted"],
                         takefocus=True, bd=0)
        self.var, self.value, self.kind, self.command = variable, value, kind, command
        s = px(self.SIZE)
        self.cv = tk.Canvas(self, width=s, height=s, bg=self["bg"], highlightthickness=0, bd=0)
        self.cv.pack(side="left", padx=(px(2), px(8)), pady=px(2))
        self.lbl = label(self, text, font)
        self.lbl.configure(bg=self["bg"])
        self.lbl.pack(side="left", pady=px(2))
        for w in (self, self.cv, self.lbl):
            w.bind("<Button-1>", self._click)
        self.bind("<space>", self._click)
        self.bind("<Return>", self._click)
        self._trace = variable.trace_add("write", lambda *_: self._draw())
        self.bind("<Destroy>", self._gone)
        self._draw()

    def _gone(self, _e) -> None:
        try:
            self.var.trace_remove("write", self._trace)
        except (tk.TclError, ValueError, RuntimeError):
            pass

    def _on(self) -> bool:
        try:
            v = self.var.get()
        except tk.TclError:
            return False
        return bool(v) if self.kind == "check" else v == self.value

    def _click(self, _e=None) -> str:
        if self.kind == "check":
            self.var.set(not self._on())
        else:
            self.var.set(self.value)
        if self.command:
            self.command()
        return "break"

    def _draw(self) -> None:
        if not self.winfo_exists():
            return
        s, on = px(self.SIZE), self._on()
        c = self.cv
        c.delete("all")
        edge = T["accent"] if on else T["muted"]
        if self.kind == "check":
            c.create_rectangle(1, 1, s - 1, s - 1, outline=edge, fill=T["card2"], width=1)
            if on:
                m = px(4)
                c.create_rectangle(m, m, s - m, s - m, outline=T["accent"], fill=T["accent"])
        else:
            c.create_oval(1, 1, s - 1, s - 1, outline=edge, fill=T["card2"], width=1)
            if on:
                m = px(4)
                c.create_oval(m, m, s - m, s - m, outline=T["accent"], fill=T["accent"])


# ---------------------------------------------------------------------- tabs
class Tabs(tk.Frame):
    """Monospaced tab labels with a white underline on the active one."""

    def __init__(self, parent: tk.Misc, items: list[tuple[str, str]], command) -> None:
        super().__init__(parent, bg=_bg_of(parent))
        self.command = command
        self.items = {}
        self.active = ""
        for key, text in items:
            cell = tk.Frame(self, bg=self["bg"])
            cell.pack(side="left", padx=(0, px(22)))
            lbl = tk.Label(cell, text=text.upper(), font=F["cap_b"], fg=T["muted"], bg=self["bg"],
                           cursor="hand2", bd=0, pady=px(6))
            lbl.pack()
            line = tk.Frame(cell, bg=self["bg"], height=px(2))
            line.pack(fill="x")
            for w in (cell, lbl):
                w.bind("<Button-1>", lambda _e, k=key: self.select(k))
            lbl.bind("<Enter>", lambda _e, k=key: self._hover(k, True))
            lbl.bind("<Leave>", lambda _e, k=key: self._hover(k, False))
            self.items[key] = (lbl, line)

    def _hover(self, key: str, on: bool) -> None:
        if key != self.active and key in self.items:
            self.items[key][0].configure(fg=T["text"] if on else T["muted"])

    def select(self, key: str, fire: bool = True) -> None:
        self.active = key
        for k, (lbl, line) in self.items.items():
            lbl.configure(fg=T["accent"] if k == key else T["muted"])
            line.configure(bg=T["accent"] if k == key else self["bg"])
        if fire and self.command:
            self.command(key)


# ----------------------------------------------------------------- selectable
def make_clickable(widget: tk.Misc, command, cursor: str = "hand2") -> None:
    """Make a whole card (and everything inside it) respond to a click.

    A ``Toggle`` already handles its own click, so it (and what is inside it) is
    left alone - binding it twice would flip it back.
    """
    if isinstance(widget, Toggle):
        return
    try:
        widget.bind("<Button-1>", lambda _e: command(), add="+")
        widget.configure(cursor=cursor)
    except tk.TclError:
        pass
    for child in widget.winfo_children():
        make_clickable(child, command, cursor)
