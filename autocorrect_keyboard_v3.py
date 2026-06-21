"""
Autocorrect Studio — with N-gram Next-Word Prediction
======================================================
"""

import tkinter as tk
from tkinter import ttk
import difflib
import os
import pickle
import threading
from collections import Counter, defaultdict
from spellchecker import SpellChecker
from textblob import Word

# ── Palette ───────────────────────────────────────────────
BG         = "#0F172A"
SURFACE    = "#1E293B"
SURFACE2   = "#263245"
BORDER     = "#334155"
ACCENT     = "#6366F1"
ACCENT_HOV = "#818CF8"
SUCCESS    = "#10B981"
ERROR_CLR  = "#F43F5E"
WARNING    = "#F59E0B"
TEXT_PRI   = "#F1F5F9"
TEXT_SEC   = "#94A3B8"
TEXT_MUTED = "#475569"

FONT_UI    = ("Segoe UI", 10)
FONT_UI_B  = ("Segoe UI", 10, "bold")
FONT_HEAD  = ("Segoe UI", 22, "bold")
FONT_SUB   = ("Segoe UI", 11)
FONT_MONO  = ("Consolas", 11)
FONT_BADGE = ("Segoe UI", 8, "bold")
FONT_LABEL = ("Segoe UI", 9)
FONT_PRED  = ("Segoe UI", 10, "bold")

CACHE_PATH = os.path.join(os.path.expanduser("~"), ".autocorrect_ngram_cache.pkl")


# ══════════════════════════════════════════════════════════
#  N-GRAM MODEL
# ══════════════════════════════════════════════════════════

class NgramPredictor:

    def __init__(self):
        self.bigram_freq:  dict = {}
        self.trigram_freq: dict = {}
        self.top_unigrams: list = []
        self.ready = False

    def build(self, on_progress=None):
        def _report(msg):
            if on_progress:
                on_progress(msg)

        # ── Try loading from cache first ──────────────────
        if os.path.exists(CACHE_PATH):
            try:
                _report("Loading saved model…")
                with open(CACHE_PATH, "rb") as f:
                    data = pickle.load(f)
                self.bigram_freq  = data["bigram"]
                self.trigram_freq = data["trigram"]
                self.top_unigrams = data["unigrams"]
                self.ready = True
                _report("Model ready ✓")
                return
            except Exception:
                pass  # corrupted cache — rebuild below

        # ── Build from corpus ─────────────────────────────
        _report("Downloading NLTK corpus (first run)…")
        import nltk
        nltk.download("brown",     quiet=True)
        nltk.download("punkt",     quiet=True)
        nltk.download("punkt_tab", quiet=True)

        from nltk.corpus import brown
        from nltk.util import ngrams

        _report("Tokenising corpus…")
        words = [w.lower() for w in brown.words() if w.isalpha()]

        _report("Building bigram table…")
        bf: dict = defaultdict(Counter)
        for w1, w2 in ngrams(words, 2):
            bf[w1][w2] += 1

        _report("Building trigram table…")
        tf: dict = defaultdict(Counter)
        for w1, w2, w3 in ngrams(words, 3):
            tf[(w1, w2)][w3] += 1

        _report("Pruning low-frequency entries…")
        # Keep top-5 completions per context; drop contexts seen < 2×
        self.bigram_freq  = {
            k: Counter(dict(v.most_common(5)))
            for k, v in bf.items() if sum(v.values()) >= 2
        }
        self.trigram_freq = {
            k: Counter(dict(v.most_common(5)))
            for k, v in tf.items() if sum(v.values()) >= 2
        }

        # Global unigram fallback
        uni: Counter = Counter(words)
        self.top_unigrams = [w for w, _ in uni.most_common(20)]

        _report("Caching model to disk…")
        try:
            with open(CACHE_PATH, "wb") as f:
                pickle.dump({
                    "bigram":   self.bigram_freq,
                    "trigram":  self.trigram_freq,
                    "unigrams": self.top_unigrams,
                }, f)
        except Exception:
            pass  # non-fatal — model still works in memory

        self.ready = True
        _report("Model ready ✓")

    def predict(self, context_words: list[str], n: int = 4) -> list[str]:
        if not self.ready:
            return []

        ctx = [w.lower() for w in context_words if w.isalpha()]

        # 1. Trigram
        if len(ctx) >= 2:
            key = (ctx[-2], ctx[-1])
            if key in self.trigram_freq:
                return [w for w, _ in self.trigram_freq[key].most_common(n)]

        # 2. Bigram
        if ctx:
            key = ctx[-1]
            if key in self.bigram_freq:
                return [w for w, _ in self.bigram_freq[key].most_common(n)]

        # 3. Unigram fallback
        return self.top_unigrams[:n]


# ══════════════════════════════════════════════════════════
#  SPELL-CHECKER HELPERS
# ══════════════════════════════════════════════════════════

spell = SpellChecker()
misspelled_data:     dict = {}
current_text_words:  list = []
_selected_listbox_index    = None
predictor = NgramPredictor()


def get_merged_suggestions(clean_word: str) -> list[str]:
    suggestions, seen = [], set()

    def add(w):
        if w and w not in seen:
            seen.add(w); suggestions.append(w)

    best_py = spell.correction(clean_word.lower())
    if best_py:
        add(best_py)

    try:
        for s, _ in Word(clean_word.lower()).spellcheck():
            add(s)
    except Exception:
        pass

    for s in difflib.get_close_matches(
        clean_word.lower(), spell.word_frequency.words(), n=8, cutoff=0.6
    ):
        add(s)

    for s in (spell.candidates(clean_word.lower()) or []):
        add(s)

    if not suggestions:
        add(clean_word)
    return suggestions


def highlight_errors():
    input_text.tag_remove("misspelled", "1.0", tk.END)
    for data in misspelled_data.values():
        words_before  = current_text_words[: data["index"]]
        char_offset   = sum(len(w) + 1 for w in words_before)
        start = f"1.{char_offset}"
        end   = f"1.{char_offset + len(data['original_word'])}"
        input_text.tag_add("misspelled", start, end)
    input_text.tag_config("misspelled", foreground=ERROR_CLR, underline=True)


def update_corrected_text():
    corrected_output.config(state=tk.NORMAL)
    corrected_output.delete("1.0", tk.END)
    corrected_output.insert(tk.END, " ".join(current_text_words))
    corrected_output.config(state=tk.DISABLED)


def update_stats():
    total    = len(current_text_words)
    errors   = len(misspelled_data)
    accuracy = ((total - errors) / total * 100) if total else 100.0

    words_var.set(str(total))
    errors_var.set(str(errors))
    accuracy_var.set(f"{accuracy:.1f}%")

    accuracy_val_label.config(fg=SUCCESS if accuracy >= 95 else (WARNING if accuracy >= 80 else ERROR_CLR))
    errors_val_label.config(fg=(ERROR_CLR if errors else SUCCESS))
    badge_count.set(f" {errors} " if errors else "")


# ══════════════════════════════════════════════════════════
#  NEXT-WORD PREDICTION  (live, as-you-type)
# ══════════════════════════════════════════════════════════

_after_id = None   # debounce timer handle

def refresh_predictions(event=None):
    global _after_id
    if _after_id:
        root.after_cancel(_after_id)
    _after_id = root.after(150, _do_predict)


def _is_complete_word(word: str) -> bool:
    return word.lower() in predictor.bigram_freq or len(word) <= 2


def _do_predict():
    if not predictor.ready:
        return

    raw = input_text.get("1.0", tk.END).strip()
    if not raw:
        _clear_predictions()
        return

    tokens = [w for w in raw.split() if w.isalpha()]

    # Only strip the last token if the user is genuinely mid-word.
    # A token that already exists in our vocabulary is considered complete —
    # e.g. 'the' in "going to the" should stay as context, not be stripped.
    if raw and raw[-1] != " " and tokens:
        if not _is_complete_word(tokens[-1]):
            tokens = tokens[:-1]   # drop true partial (e.g. 'uni', 'qu')

    if not tokens:
        _clear_predictions()
        return

    preds = predictor.predict(tokens, n=4)
    _render_predictions(preds)


def _clear_predictions():
    for btn in pred_buttons:
        btn.config(text="", state=tk.DISABLED, bg=SURFACE2)


def _render_predictions(preds: list[str]):
    for i, btn in enumerate(pred_buttons):
        if i < len(preds):
            word = preds[i]
            btn.config(
                text=word,
                state=tk.NORMAL,
                bg=SURFACE2,
                command=lambda w=word: _insert_prediction(w),
            )
        else:
            btn.config(text="", state=tk.DISABLED, bg=SURFACE2)


def _insert_prediction(word: str):
    raw = input_text.get("1.0", tk.END)
    # If line ends without a space, add one before the word
    if raw.strip() and not raw.rstrip("\n").endswith(" "):
        input_text.insert(tk.INSERT, " ")
    input_text.insert(tk.INSERT, word + " ")
    input_text.see(tk.INSERT)
    refresh_predictions()
    set_status(f'Inserted "{word}". Keep typing or click another prediction.', SUCCESS)


# ══════════════════════════════════════════════════════════
#  SPELL-CHECK ACTIONS
# ══════════════════════════════════════════════════════════

def check_text():
    global misspelled_data, current_text_words, _selected_listbox_index

    text = input_text.get("1.0", tk.END).strip()
    if not text:
        set_status("Paste or type text above, then click Check.", TEXT_MUTED)
        return

    current_text_words = text.split()
    misspelled_data.clear()
    _selected_listbox_index = None
    misspelled_list.delete(0, tk.END)
    suggestion_list.delete(0, tk.END)

    for i, word in enumerate(current_text_words):
        clean = word.strip(".,!?;:()\"'")
        if not clean:
            continue
        if clean.lower() not in spell:
            suggestions = get_merged_suggestions(clean)
            key = f"{i}_{word}"
            misspelled_data[key] = {
                "index": i,
                "original_word": word,
                "clean_word": clean,
                "suggestions": suggestions,
            }
            misspelled_list.insert(tk.END, f"  {word}")

    highlight_errors()
    update_corrected_text()
    update_stats()
    _color_rows(misspelled_list)

    msg = (f"Found {len(misspelled_data)} issue(s). Click a word → pick a suggestion → Apply."
           if misspelled_data else "All good — no spelling errors found!")
    set_status(msg, WARNING if misspelled_data else SUCCESS)


def show_suggestions(event=None):
    global _selected_listbox_index
    sel = misspelled_list.curselection()
    if not sel:
        return
    _selected_listbox_index = sel[0]
    suggestion_list.delete(0, tk.END)

    keys = list(misspelled_data.keys())
    if _selected_listbox_index < len(keys):
        key  = keys[_selected_listbox_index]
        data = misspelled_data[key]
        for s in data["suggestions"][:15]:
            suggestion_list.insert(tk.END, f"  {s}")
        if data["suggestions"]:
            suggestion_list.selection_set(0)
            suggestion_list.activate(0)
        _color_rows(suggestion_list)
        set_status(
            f'Suggestions for "{data["original_word"]}" — double-click or Enter to apply.',
            TEXT_SEC,
        )


def _active_suggestion_idx():
    sel = suggestion_list.curselection()
    if sel:
        return sel[0]
    a = suggestion_list.index(tk.ACTIVE)
    return a if a is not None and a >= 0 else None


def apply_suggestion(event=None):
    global _selected_listbox_index
    if _selected_listbox_index is None:
        set_status("Select a misspelled word first.", ERROR_CLR); return
    sug_idx = _active_suggestion_idx()
    if sug_idx is None:
        set_status("Select a suggestion.", ERROR_CLR); return

    keys = list(misspelled_data.keys())
    if _selected_listbox_index >= len(keys):
        return

    key  = keys[_selected_listbox_index]
    data = misspelled_data[key]
    replacement = suggestion_list.get(sug_idx).strip()
    final_word  = data["original_word"].replace(data["clean_word"], replacement)

    current_text_words[data["index"]] = final_word
    del misspelled_data[key]
    misspelled_list.delete(_selected_listbox_index)
    suggestion_list.delete(0, tk.END)
    _selected_listbox_index = None

    highlight_errors()
    update_corrected_text()
    update_stats()
    _color_rows(misspelled_list)
    _flash_success()
    set_status(f'✓  Replaced with "{replacement}".', SUCCESS)


def apply_all():
    global _selected_listbox_index
    count = len(misspelled_data)
    for key in list(misspelled_data.keys()):
        data = misspelled_data[key]
        best = data["suggestions"][0]
        final = data["original_word"].replace(data["clean_word"], best)
        current_text_words[data["index"]] = final
        del misspelled_data[key]

    misspelled_list.delete(0, tk.END)
    suggestion_list.delete(0, tk.END)
    _selected_listbox_index = None

    highlight_errors()
    update_corrected_text()
    update_stats()
    _flash_success()
    set_status(f"✓  Auto-corrected {count} word(s).", SUCCESS)


def copy_corrected():
    text = corrected_output.get("1.0", tk.END).strip()
    if text:
        root.clipboard_clear()
        root.clipboard_append(text)
        set_status("✓  Corrected text copied to clipboard.", SUCCESS)


def clear_all():
    global current_text_words, _selected_listbox_index
    input_text.delete("1.0", tk.END)
    input_text.tag_remove("misspelled", "1.0", tk.END)
    corrected_output.config(state=tk.NORMAL)
    corrected_output.delete("1.0", tk.END)
    corrected_output.config(state=tk.DISABLED)
    misspelled_list.delete(0, tk.END)
    suggestion_list.delete(0, tk.END)
    misspelled_data.clear()
    current_text_words = []
    _selected_listbox_index = None
    words_var.set("0"); errors_var.set("0"); accuracy_var.set("100%")
    accuracy_val_label.config(fg=SUCCESS)
    errors_val_label.config(fg=TEXT_SEC)
    badge_count.set("")
    _clear_predictions()
    set_status("Cleared. Type new text to begin.", TEXT_MUTED)


# ── Visual helpers ────────────────────────────────────────

def set_status(msg, color=TEXT_SEC):
    status_var.set(msg)
    status_label.config(fg=color)


def _flash_success():
    corrected_frame.config(highlightbackground=SUCCESS, highlightthickness=2)
    root.after(600, lambda: corrected_frame.config(
        highlightbackground=BORDER, highlightthickness=1
    ))


def _color_rows(lb):
    for i in range(lb.size()):
        bg = SURFACE if i % 2 == 0 else SURFACE2
        lb.itemconfig(i, bg=bg, fg=TEXT_PRI,
                      selectbackground=ACCENT, selectforeground="#FFFFFF")


# ── Widget factories ──────────────────────────────────────

def make_button(parent, text, cmd, bg=ACCENT, fg="#FFFFFF", width=None):
    cfg = dict(
        text=text, command=cmd,
        bg=bg, fg=fg, activebackground=ACCENT_HOV, activeforeground="#FFFFFF",
        font=FONT_UI_B, relief=tk.FLAT, bd=0,
        padx=16, pady=8, cursor="hand2",
    )
    if width:
        cfg["width"] = width
    btn = tk.Button(parent, **cfg)
    btn.bind("<Enter>", lambda e: btn.config(bg=ACCENT_HOV if bg == ACCENT else bg))
    btn.bind("<Leave>", lambda e: btn.config(bg=bg))
    return btn


def styled_listbox(parent):
    frame = tk.Frame(parent, bg=SURFACE)
    frame.pack(fill=tk.BOTH, expand=True)
    sb = tk.Scrollbar(frame, orient=tk.VERTICAL, bg=SURFACE,
                      troughcolor=BG, relief=tk.FLAT, bd=0)
    lb = tk.Listbox(
        frame, font=FONT_UI,
        bg=SURFACE, fg=TEXT_PRI,
        relief=tk.FLAT, bd=0,
        selectbackground=ACCENT, selectforeground="#FFFFFF",
        activestyle="none",
        yscrollcommand=sb.set,
        highlightthickness=0,
        exportselection=False,
        cursor="hand2",
    )
    sb.config(command=lb.yview)
    lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    return lb


def panel(parent, col, col_weight=1):
    f = tk.Frame(parent, bg=SURFACE,
                 highlightbackground=BORDER, highlightthickness=1)
    f.grid(row=0, column=col, sticky="nsew",
           padx=(0, 10) if col < 3 else (0, 0), pady=(0, 16))
    return f


def panel_header(parent, title_text, badge_var=None):
    hdr = tk.Frame(parent, bg=BORDER, padx=14, pady=10)
    hdr.pack(fill=tk.X)
    row = tk.Frame(hdr, bg=BORDER)
    row.pack(fill=tk.X)
    tk.Label(row, text=title_text.upper(),
             font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=BORDER).pack(side=tk.LEFT)
    if badge_var is not None:
        tk.Label(row, textvariable=badge_var,
                 font=FONT_BADGE, fg="#FFFFFF", bg=ERROR_CLR,
                 padx=4, pady=1).pack(side=tk.LEFT, padx=6)


# ══════════════════════════════════════════════════════════
#  BUILD GUI
# ══════════════════════════════════════════════════════════

root = tk.Tk()
root.title("Autocorrect Studio — N-gram Prediction")
root.geometry("1300x900")
root.configure(bg=BG)
root.resizable(True, True)

# ── Header ────────────────────────────────────────────────
header_outer = tk.Frame(root, bg=SURFACE)
header_outer.pack(fill=tk.X)
header_inner = tk.Frame(header_outer, bg=SURFACE, padx=28, pady=14)
header_inner.pack(fill=tk.X)

tk.Label(header_inner, text="Autocorrect Studio",
         font=FONT_HEAD, fg=TEXT_PRI, bg=SURFACE).pack(side=tk.LEFT)
tk.Label(header_inner, text="Spell correction + N-gram next-word prediction",
         font=FONT_SUB, fg=TEXT_MUTED, bg=SURFACE, padx=12).pack(side=tk.LEFT, pady=(6, 0))

tk.Frame(root, bg=ACCENT, height=2).pack(fill=tk.X)

# ── Stat cards ────────────────────────────────────────────
stats_row = tk.Frame(root, bg=BG, padx=24, pady=14)
stats_row.pack(fill=tk.X)

words_var    = tk.StringVar(value="0")
errors_var   = tk.StringVar(value="0")
accuracy_var = tk.StringVar(value="100%")
badge_count  = tk.StringVar(value="")

errors_val_label   = None
accuracy_val_label = None

for label, var, is_errors, is_accuracy in [
    ("TOTAL WORDS",  words_var,    False, False),
    ("ERRORS FOUND", errors_var,   True,  False),
    ("ACCURACY",     accuracy_var, False, True),
]:
    card = tk.Frame(stats_row, bg=SURFACE, padx=18, pady=10,
                    highlightbackground=BORDER, highlightthickness=1)
    card.pack(side=tk.LEFT, padx=(0, 12))
    tk.Label(card, text=label, fg=TEXT_MUTED, bg=SURFACE, font=FONT_LABEL).pack(anchor="w")
    val_lbl = tk.Label(card, textvariable=var, fg=TEXT_PRI, bg=SURFACE,
                       font=("Segoe UI", 20, "bold"))
    val_lbl.pack(anchor="w")
    if is_errors:
        errors_val_label = val_lbl
    if is_accuracy:
        accuracy_val_label = val_lbl
        val_lbl.config(fg=SUCCESS)

# ── Input box ─────────────────────────────────────────────
input_outer = tk.Frame(root, bg=BG, padx=24)
input_outer.pack(fill=tk.X)

tk.Label(input_outer, text="INPUT TEXT",
         font=("Segoe UI", 8, "bold"), fg=ACCENT, bg=BG).pack(anchor="w", pady=(0, 4))

input_frame = tk.Frame(input_outer, bg=SURFACE,
                       highlightbackground=BORDER, highlightthickness=1)
input_frame.pack(fill=tk.X)

input_text = tk.Text(
    input_frame, height=5, font=FONT_MONO,
    bg=SURFACE, fg=TEXT_PRI, insertbackground=TEXT_PRI,
    relief=tk.FLAT, bd=0, padx=14, pady=10,
    selectbackground=ACCENT, selectforeground="#FFFFFF",
    wrap=tk.WORD,
)
input_text.pack(fill=tk.X)
input_text.bind("<FocusIn>",  lambda e: input_frame.config(highlightbackground=ACCENT, highlightthickness=2))
input_text.bind("<FocusOut>", lambda e: input_frame.config(highlightbackground=BORDER, highlightthickness=1))
input_text.bind("<KeyRelease>", refresh_predictions)

# ── Prediction bar ────────────────────────────────────────
pred_outer = tk.Frame(root, bg=BG, padx=24, pady=0)
pred_outer.pack(fill=tk.X)

pred_label_frame = tk.Frame(pred_outer, bg=BG)
pred_label_frame.pack(fill=tk.X, pady=(6, 4))

tk.Label(pred_label_frame, text="NEXT WORD",
         font=("Segoe UI", 8, "bold"), fg=ACCENT, bg=BG).pack(side=tk.LEFT)

# Model status label (shown while loading)
model_status_var = tk.StringVar(value="⏳ Loading prediction model…")
model_status_lbl = tk.Label(pred_label_frame, textvariable=model_status_var,
                             font=FONT_LABEL, fg=WARNING, bg=BG)
model_status_lbl.pack(side=tk.LEFT, padx=10)

pred_bar = tk.Frame(pred_outer, bg=SURFACE2,
                    highlightbackground=BORDER, highlightthickness=1)
pred_bar.pack(fill=tk.X)

pred_buttons: list[tk.Button] = []
for i in range(4):
    btn = tk.Button(
        pred_bar, text="", font=FONT_PRED,
        bg=SURFACE2, fg=TEXT_PRI,
        activebackground=ACCENT, activeforeground="#FFFFFF",
        relief=tk.FLAT, bd=0, padx=20, pady=10,
        state=tk.DISABLED, cursor="hand2",
    )
    btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
    # Separator between buttons
    if i < 3:
        tk.Frame(pred_bar, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, pady=6)
    pred_buttons.append(btn)

    def _hover_in(e, b=btn):
        if b["state"] != tk.DISABLED:
            b.config(bg=ACCENT, fg="#FFFFFF")
    def _hover_out(e, b=btn):
        if b["state"] != tk.DISABLED:
            b.config(bg=SURFACE2, fg=TEXT_PRI)
    btn.bind("<Enter>", _hover_in)
    btn.bind("<Leave>", _hover_out)

tk.Label(pred_outer, text="↑ Click a word to insert it at your cursor position",
         font=FONT_LABEL, fg=TEXT_MUTED, bg=BG).pack(anchor="w", pady=(3, 0))

# ── Toolbar ───────────────────────────────────────────────
toolbar = tk.Frame(root, bg=BG, padx=24, pady=8)
toolbar.pack(fill=tk.X)

make_button(toolbar, "⟳  Check Spelling", check_text).pack(side=tk.LEFT, padx=(0, 8))
make_button(toolbar, "✦  Auto-fix All", apply_all, bg="#0F766E").pack(side=tk.LEFT, padx=(0, 8))
make_button(toolbar, "⎘  Copy Result", copy_corrected, bg="#7C3AED").pack(side=tk.LEFT, padx=(0, 8))
make_button(toolbar, "✕  Clear", clear_all, bg=SURFACE, fg=TEXT_SEC).pack(side=tk.LEFT)
root.bind("<Control-Return>", lambda e: check_text())

# ── Status bar ────────────────────────────────────────────
status_var = tk.StringVar(value="Type something — predictions appear as you go.")
status_label = tk.Label(root, textvariable=status_var,
                        fg=TEXT_MUTED, bg=BG, font=FONT_LABEL, anchor="w", padx=24)
status_label.pack(fill=tk.X, pady=(2, 4))

# ── Three panels ──────────────────────────────────────────
panels_outer = tk.Frame(root, bg=BG, padx=24)
panels_outer.pack(fill=tk.BOTH, expand=True)

panels_outer.columnconfigure(0, weight=1)
panels_outer.columnconfigure(1, weight=1)
panels_outer.columnconfigure(2, weight=2)
panels_outer.rowconfigure(0, weight=1)

# Panel 1 — misspelled
p1 = panel(panels_outer, 0)
panel_header(p1, "Misspelled Words", badge_count)
misspelled_list = styled_listbox(p1)
misspelled_list.bind("<<ListboxSelect>>", show_suggestions)

# Panel 2 — suggestions
p2 = panel(panels_outer, 1)
panel_header(p2, "Suggestions")
suggestion_list = styled_listbox(p2)
suggestion_list.bind("<Double-Button-1>", apply_suggestion)
suggestion_list.bind("<Return>",          apply_suggestion)

btn_apply_wrap = tk.Frame(p2, bg=SURFACE, padx=10, pady=10)
btn_apply_wrap.pack(fill=tk.X)
make_button(btn_apply_wrap, "✔  Apply Selected", apply_suggestion, width=20).pack(fill=tk.X)
tk.Label(btn_apply_wrap, text="or double-click · Enter key",
         font=FONT_LABEL, fg=TEXT_MUTED, bg=SURFACE).pack(pady=(4, 0))

# Panel 3 — corrected output
p3 = panel(panels_outer, 2)
panel_header(p3, "Corrected Output")

corrected_frame = tk.Frame(p3, bg=SURFACE,
                           highlightbackground=BORDER, highlightthickness=1)
corrected_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

corrected_output = tk.Text(
    corrected_frame, font=FONT_MONO,
    bg=SURFACE, fg=SUCCESS,
    relief=tk.FLAT, bd=0, padx=12, pady=10,
    selectbackground=ACCENT, selectforeground="#FFFFFF",
    wrap=tk.WORD, state=tk.DISABLED, highlightthickness=0,
)
corrected_output.pack(fill=tk.BOTH, expand=True)

# ── Footer ────────────────────────────────────────────────
tk.Frame(root, bg=SURFACE, pady=0).pack(fill=tk.X, side=tk.BOTTOM)
tk.Label(root,
         text="Developed by Pranav  ·  ShadowFox AIML Internship  ·  Ctrl+Enter to check spelling",
         font=FONT_LABEL, fg=TEXT_MUTED, bg=SURFACE).pack(side=tk.BOTTOM, pady=6)

# ══════════════════════════════════════════════════════════
#  BACKGROUND MODEL LOADING  (non-blocking)
# ══════════════════════════════════════════════════════════

def _load_model_thread():
    def on_progress(msg):
        # Update UI from the main thread
        root.after(0, lambda m=msg: model_status_var.set(f"⏳ {m}"))

    predictor.build(on_progress=on_progress)

    def on_done():
        model_status_var.set("✓ Prediction model ready")
        model_status_lbl.config(fg=SUCCESS)
        # Trigger an initial prediction if text is already in the box
        refresh_predictions()

    root.after(0, on_done)


threading.Thread(target=_load_model_thread, daemon=True).start()

root.mainloop()