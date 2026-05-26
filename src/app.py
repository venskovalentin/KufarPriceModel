import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime


from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QComboBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QFrame, QScrollArea, QSpinBox,
    QDoubleSpinBox, QLineEdit, QProgressBar, QSizePolicy, QGridLayout,
    QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ── colours ──────────────────────────────────────────────────────────────────
BG       = "#1e1e2e"
BG_DEEP  = "#15151f"
PANEL    = "#2a2a3e"
PANEL2   = "#313146"
BORDER   = "#3a3a52"
TEXT     = "#e6e6f0"
TEXT_DIM = "#9b9bb4"
TEXT_FAINT = "#6c6c84"
ACCENT   = "#4f9cf9"
GREEN    = "#3dd68c"
ORANGE   = "#f9a84f"
RED      = "#f96e6e"

QSS = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: "Segoe UI", "Inter", system-ui;
    font-size: 13px;
}}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{
    background: {PANEL};
    color: {TEXT_DIM};
    padding: 8px 20px;
    border: 1px solid {BORDER};
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background: {BG};
    color: {TEXT};
    border-bottom: 2px solid {ACCENT};
}}
QComboBox {{
    background: #1a1a26;
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 10px;
    color: {TEXT};
    min-height: 28px;
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {PANEL2};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    color: {TEXT};
}}
QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: #1a1a26;
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 10px;
    color: {TEXT};
    min-height: 28px;
}}
QPushButton {{
    background: {ACCENT};
    color: white;
    border: none;
    border-radius: 7px;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 13px;
}}
QPushButton:hover {{ background: #6aaefb; }}
QPushButton:disabled {{ background: #3a3a52; color: {TEXT_DIM}; }}
QPushButton#ghost {{
    background: #1a1a26;
    color: {TEXT};
    border: 1px solid {BORDER};
}}
QPushButton#ghost:hover {{ background: {PANEL2}; }}
QPushButton#green {{
    background: {GREEN};
    color: #0d1f17;
}}
QPushButton#green:hover {{ background: #50e09c; }}
QTableWidget {{
    background: {PANEL};
    border: none;
    gridline-color: {BORDER};
    color: {TEXT};
    font-size: 12px;
}}
QTableWidget::item {{ padding: 6px 10px; border-bottom: 1px solid {BORDER}; }}
QTableWidget::item:selected {{ background: #1e3a5f; color: {TEXT}; }}
QHeaderView::section {{
    background: #1f1f2c;
    color: {TEXT_FAINT};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 7px 10px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}}
QScrollBar:vertical {{
    background: {BG_DEEP};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QProgressBar {{
    background: #1a1a26;
    border: 1px solid {BORDER};
    border-radius: 5px;
    height: 8px;
    text-align: center;
    font-size: 11px;
    color: {TEXT_DIM};
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT}, stop:1 #7b6cf9);
    border-radius: 4px;
}}
QLabel#card_val {{
    font-size: 28px;
    font-weight: 700;
    color: {TEXT};
}}
QLabel#section_title {{
    font-size: 13px;
    font-weight: 600;
    color: {TEXT};
}}
QLabel#muted {{
    color: {TEXT_DIM};
    font-size: 12px;
}}
QFrame#card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
QFrame#sidebar {{
    background: #1c1c2a;
    border-right: 1px solid {BORDER};
}}
QFrame#topbar {{
    background: #22222f;
    border-bottom: 1px solid {BORDER};
}}
"""


def card_frame(parent=None):
    f = QFrame(parent)
    f.setObjectName("card")
    return f


def label(text, color=TEXT, size=13, bold=False, parent=None):
    l = QLabel(text, parent)
    style = f"color: {color}; font-size: {size}px;"
    if bold:
        style += " font-weight: 600;"
    l.setStyleSheet(style)
    return l


# ── Matplotlib canvas ─────────────────────────────────────────────────────────
class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=3):
        self.fig = Figure(figsize=(width, height), facecolor=BG)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(BG)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


# ── Retrain worker ────────────────────────────────────────────────────────────
class RetrainWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def run(self):
        try:
            # import here to avoid circular issues at module load
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from retrainer import retrainer as do_retrain

            original_print = __builtins__["print"] if isinstance(__builtins__, dict) else print

            page = [0]
            def progress_print(*args, **kwargs):
                msg = " ".join(str(a) for a in args)
                if "processing page" in msg.lower():
                    try:
                        n = int(msg.strip().split()[-1])
                        page[0] = n
                        pct = min(int(n / 6), 99)  # ~588 pages
                        self.progress.emit(pct, f"Parsing page {n}…")
                    except Exception:
                        pass
                original_print(*args, **kwargs)

            import builtins
            builtins.print = progress_print

            pipeline = do_retrain(log_target=True)

            builtins.print = original_print
            self.progress.emit(100, "Training complete")

            meta_path = os.path.join(os.path.dirname(__file__), "..", "models", "model_meta.json")
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            self.finished.emit(meta)
        except Exception as e:
            self.error.emit(str(e))


# ── Tab 1: Market Analysis ────────────────────────────────────────────────────
class TabMarket(QWidget):
    def __init__(self, analyzer, predictor):
        super().__init__()
        self.analyzer = analyzer
        self.predictor = predictor
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── sidebar filters ──────────────────────────────────────────────────
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(240)
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(16, 16, 16, 16)
        sb_layout.setSpacing(14)

        sb_layout.addWidget(label("Filters", bold=True))

        self.f_brand = QComboBox()
        self.f_proc  = QComboBox()
        self.f_cond  = QComboBox()
        self.f_rom   = QComboBox()

        for lbl_text, widget in [
            ("Brand", self.f_brand),
            ("Processor", self.f_proc),
            ("Condition", self.f_cond),
            ("Storage type", self.f_rom),
        ]:
            grp = QVBoxLayout()
            grp.setSpacing(4)
            grp.addWidget(label(lbl_text, color=TEXT_FAINT, size=11))
            grp.addWidget(widget)
            sb_layout.addLayout(grp)

        self.price_min = QSpinBox()
        self.price_max = QSpinBox()
        for sp in (self.price_min, self.price_max):
            sp.setRange(0, 99999)
            sp.setSingleStep(100)
        self.price_min.setValue(0)
        self.price_max.setValue(10000)

        price_grp = QVBoxLayout()
        price_grp.setSpacing(4)
        price_grp.addWidget(label("Price range (BYN)", color=TEXT_FAINT, size=11))
        price_row = QHBoxLayout()
        price_row.addWidget(self.price_min)
        price_row.addWidget(label("—", color=TEXT_DIM))
        price_row.addWidget(self.price_max)
        price_grp.addLayout(price_row)
        sb_layout.addLayout(price_grp)

        btn_apply = QPushButton("Apply filters")
        btn_apply.clicked.connect(self._refresh)
        sb_layout.addWidget(btn_apply)

        self.btn_underpriced = QPushButton("Find underpriced")
        self.btn_underpriced.setObjectName("green")
        self.btn_underpriced.clicked.connect(self._find_underpriced)
        sb_layout.addWidget(self.btn_underpriced)

        sb_layout.addStretch()

        self.stats_label = label("", color=TEXT_DIM, size=11)
        self.stats_label.setWordWrap(True)
        sb_layout.addWidget(self.stats_label)

        root.addWidget(sidebar)

        # ── main area ────────────────────────────────────────────────────────
        main = QVBoxLayout()
        main.setContentsMargins(18, 18, 18, 18)
        main.setSpacing(14)

        # histogram
        self.canvas = MplCanvas(height=2.8)
        hist_card = card_frame()
        hc_layout = QVBoxLayout(hist_card)
        hc_layout.setContentsMargins(12, 12, 12, 8)
        hc_layout.addWidget(self.canvas)
        main.addWidget(hist_card, stretch=2)

        # table
        table_card = card_frame()
        tc_layout = QVBoxLayout(table_card)
        tc_layout.setContentsMargins(0, 0, 0, 0)
        tc_layout.setSpacing(0)

        thead = QHBoxLayout()
        thead.setContentsMargins(14, 10, 14, 10)
        self.table_title = label("Listings", bold=True)
        thead.addWidget(self.table_title)
        thead.addStretch()
        tc_layout.addLayout(thead)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"color: {BORDER};")
        tc_layout.addWidget(line)

        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(
            ["Subject", "Brand", "Price BYN", "Processor", "RAM", "ROM", "Diagonal", "Condition"]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, 8):
            self.table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            self.table.styleSheet() +
            f"QTableWidget {{ alternate-background-color: {PANEL2}; }}"
        )
        tc_layout.addWidget(self.table)
        main.addWidget(table_card, stretch=3)

        main_widget = QWidget()
        main_widget.setLayout(main)
        root.addWidget(main_widget)

    def _get_filters(self):
        f = {}
        if self.f_brand.currentIndex() > 0:
            f["brand"] = self.f_brand.currentText()
        if self.f_proc.currentIndex() > 0:
            f["processor"] = self.f_proc.currentText()
        if self.f_cond.currentIndex() > 0:
            f["condition"] = self.f_cond.currentText()
        if self.f_rom.currentIndex() > 0:
            f["rom_type"] = self.f_rom.currentText()
        return f

    def _refresh(self):
        if not self.analyzer:
            return

        df = self.analyzer.df
        filters = self._get_filters()
        if filters:
            df = self.analyzer._apply_filters(df, filters)

        p_min, p_max = self.price_min.value(), self.price_max.value()
        df = df[(df["price_byn"] >= p_min) & (df["price_byn"] <= p_max)]

        # populate filter dropdowns on first run
        if self.f_brand.count() == 0:
            self._populate_combos()

        # stats
        stats = {
            "median": round(df["price_byn"].median(), 0),
            "mean":   round(df["price_byn"].mean(), 0),
            "count":  len(df),
            "min":    round(df["price_byn"].min(), 0),
            "max":    round(df["price_byn"].max(), 0),
        }
        self.stats_label.setText(
            f"Median: {stats['median']} BYN\n"
            f"Mean: {stats['mean']} BYN\n"
            f"Count: {stats['count']}\n"
            f"Range: {stats['min']} – {stats['max']}"
        )
        self.table_title.setText(f"Listings  ({stats['count']})")

        # histogram
        ax = self.canvas.ax
        ax.clear()
        prices = df["price_byn"].dropna()
        prices = prices[prices < prices.quantile(0.99)]
        ax.hist(prices, bins=40, color=ACCENT, alpha=0.85, edgecolor="none")
        med = prices.median()
        ax.axvline(med, color=ORANGE, linewidth=1.5, linestyle="--")
        ax.text(med, ax.get_ylim()[1] * 0.9, f" {med:.0f}", color=ORANGE, fontsize=9)
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT_FAINT, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.set_xlabel("Price (BYN)", color=TEXT_DIM, fontsize=9)
        ax.set_ylabel("Count", color=TEXT_DIM, fontsize=9)
        self.canvas.fig.tight_layout(pad=0.8)
        self.canvas.draw()

        # table
        cols = ["subject", "brand", "price_byn", "processor", "ram_volume", "rom_volume", "diagonal", "condition"]
        sub = df[cols].head(200).reset_index(drop=True)
        self.table.setRowCount(len(sub))
        for i, row in sub.iterrows():
            for j, col in enumerate(cols):
                val = row[col]
                item = QTableWidgetItem("" if pd.isna(val) else str(val))
                if col == "price_byn":
                    item.setText(f"{float(val):.0f}" if not pd.isna(val) else "")
                    item.setForeground(QColor(ACCENT))
                item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(i, j, item)

    def _populate_combos(self):
        df = self.analyzer.df
        for combo, col in [
            (self.f_brand, "brand"),
            (self.f_proc, "processor"),
            (self.f_cond, "condition"),
            (self.f_rom, "rom_type"),
        ]:
            combo.addItem("All")
            vals = sorted(df[col].dropna().unique().tolist())
            for v in vals:
                combo.addItem(str(v))

    def _find_underpriced(self):
        if not self.predictor or not self.analyzer:
            return
        try:
            result = self.analyzer.find_underpriced(self.predictor)
            if result.empty:
                return
            cols = ["subject", "brand", "price_byn", "processor", "ram_volume", "rom_volume", "diagonal", "condition"]
            self.table.setRowCount(len(result))
            self.table_title.setText(f"Underpriced listings  ({len(result)})")
            for i, (_, row) in enumerate(result.iterrows()):
                for j, col in enumerate(cols):
                    val = row.get(col, "")
                    item = QTableWidgetItem("" if pd.isna(val) else str(val))
                    if col == "price_byn":
                        item.setText(f"{float(val):.0f}" if not pd.isna(val) else "")
                        item.setForeground(QColor(GREEN))
                    item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                    self.table.setItem(i, j, item)
                # green row tint
                for j in range(self.table.columnCount()):
                    if self.table.item(i, j):
                        self.table.item(i, j).setBackground(QColor(61, 214, 140, 20))
        except Exception as e:
            self.stats_label.setText(f"Error: {e}")


# ── Tab 2: Price Prediction ───────────────────────────────────────────────────
class TabPredict(QWidget):
    def __init__(self, analyzer, predictor):
        super().__init__()
        self.analyzer  = analyzer
        self.predictor = predictor
        self._build_ui()
        if analyzer:
            self._populate_form()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(18)

        # ── left: form ───────────────────────────────────────────────────────
        form_card = card_frame()
        form_card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        fl = QVBoxLayout(form_card)
        fl.setContentsMargins(18, 16, 18, 18)
        fl.setSpacing(0)

        fl.addWidget(label("Laptop characteristics", bold=True, size=14))
        fl.addSpacing(14)

        grid = QGridLayout()
        grid.setSpacing(10)

        self.fields = {}
        form_items = [
            ("brand",        "Brand",         QComboBox),
            ("processor",    "Processor",      QComboBox),
            ("condition",    "Condition",      QComboBox),
            ("rom_type",     "Storage type",   QComboBox),
            ("os",           "OS",             QComboBox),
            ("region",       "Region",         QComboBox),
            ("videocard",    "GPU type",       QComboBox),
            ("battery_life", "Battery life",   QComboBox),
            ("ram_volume",   "RAM (GB)",       QSpinBox),
            ("rom_volume",   "ROM (GB)",       QSpinBox),
            ("diagonal",     "Diagonal (inch)",QDoubleSpinBox),
        ]
        for idx, (key, lbl_text, WidgetClass) in enumerate(form_items):
            row, col = divmod(idx, 2)
            vbox = QVBoxLayout()
            vbox.setSpacing(4)
            vbox.addWidget(label(lbl_text, color=TEXT_DIM, size=11))
            w = WidgetClass()
            if isinstance(w, QSpinBox):
                w.setRange(0, 99999)
            elif isinstance(w, QDoubleSpinBox):
                w.setRange(0, 30)
                w.setSingleStep(0.1)
                w.setValue(15.6)
            vbox.addWidget(w)
            grid.addLayout(vbox, row, col)
            self.fields[key] = w

        fl.addLayout(grid)
        fl.addSpacing(16)

        btn_predict = QPushButton("Predict price")
        btn_predict.clicked.connect(self._predict)
        fl.addWidget(btn_predict)
        fl.addStretch()

        root.addWidget(form_card, stretch=1)

        # ── right: result ────────────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(14)

        # prediction card
        pred_card = card_frame()
        pred_card.setStyleSheet(
            f"QFrame#card {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:1, "
            f"stop:0 #2d2d44, stop:1 #232336); border: 1px solid #3a3a55; border-radius: 12px; }}"
        )
        pc_layout = QVBoxLayout(pred_card)
        pc_layout.setContentsMargins(28, 24, 28, 24)
        pc_layout.setSpacing(8)

        eyebrow = label("● RECOMMENDED PRICE", color=ACCENT, size=11, bold=True)
        pc_layout.addWidget(eyebrow)

        self.price_label = QLabel("—")
        self.price_label.setStyleSheet(
            f"color: {TEXT}; font-size: 58px; font-weight: 700; letter-spacing: -2px;"
        )
        pc_layout.addWidget(self.price_label)

        self.range_label = label("Confidence range: —", color=TEXT_DIM, size=12)
        pc_layout.addWidget(self.range_label)

        self.source_label = label("Based on — listings · XGBoost", color=TEXT_FAINT, size=11)
        pc_layout.addWidget(self.source_label)

        right.addWidget(pred_card, stretch=2)

        # your price card
        your_card = card_frame()
        yc_layout = QVBoxLayout(your_card)
        yc_layout.setContentsMargins(18, 14, 18, 14)
        yc_layout.setSpacing(10)

        yc_layout.addWidget(label("Compare your price", bold=True))

        row_w = QHBoxLayout()
        self.your_price_input = QLineEdit()
        self.your_price_input.setPlaceholderText("Enter your price (BYN)…")
        self.your_price_input.textChanged.connect(self._compare_price)
        row_w.addWidget(self.your_price_input)
        yc_layout.addLayout(row_w)

        self.compare_label = label("", color=TEXT_DIM, size=13)
        yc_layout.addWidget(self.compare_label)

        right.addWidget(your_card, stretch=1)
        right.addStretch()

        right_w = QWidget()
        right_w.setLayout(right)
        root.addWidget(right_w, stretch=1)

        self._predicted_price = None

    def _populate_form(self):
        df = self.analyzer.df
        combo_cols = {
            "brand":        "brand",
            "processor":    "processor",
            "condition":    "condition",
            "rom_type":     "rom_type",
            "os":           "os",
            "region":       "region",
            "videocard":    "videocard",
            "battery_life": "battery_life",
        }
        for key, col in combo_cols.items():
            w = self.fields[key]
            if isinstance(w, QComboBox) and col in df.columns:
                w.addItem("")
                vals = sorted(df[col].dropna().unique().tolist(), key=str)
                for v in vals:
                    w.addItem(str(v))

        # sensible defaults for spinboxes
        if "ram_volume" in df.columns:
            med = pd.to_numeric(df["ram_volume"].dropna().str.extract(r'(\d+)')[0]).median()
            self.fields["ram_volume"].setValue(int(med) if not np.isnan(med) else 8)
        if "rom_volume" in df.columns:
            med = pd.to_numeric(df["rom_volume"].dropna().str.extract(r'(\d+)')[0]).median()
            self.fields["rom_volume"].setValue(int(med) if not np.isnan(med) else 512)

        meta = self.predictor.get_meta() if self.predictor else {}
        n = meta.get("n_samples", "—")
        self.source_label.setText(f"Based on {n} listings · XGBoost")

    def _collect_row(self) -> dict:
        row = {}
        for key, w in self.fields.items():
            if isinstance(w, QComboBox):
                val = w.currentText()
                row[key] = val if val else np.nan
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                row[key] = w.value()
        return row

    def _predict(self):
        if not self.predictor:
            self.price_label.setText("No model")
            print("Predictor is None")  # добавь это
            return
        try:
            row = self._collect_row()
            result = self.predictor.predict_single(row)
            self._predicted_price = result["price"]
            self.price_label.setText(f"{result['price']:,.0f} BYN")
            self.range_label.setText(
                f"Confidence range: {result['lower']:,.0f} – {result['upper']:,.0f} BYN"
            )
            self._compare_price(self.your_price_input.text())
        except Exception as e:
            self.price_label.setText("Error")
            self.range_label.setText(str(e))

    def _compare_price(self, text):
        if not self._predicted_price or not text:
            self.compare_label.setText("")
            return
        try:
            your = float(text.replace(" ", "").replace(",", "."))
            diff_pct = (your - self._predicted_price) / self._predicted_price * 100
            if diff_pct < -2:
                self.compare_label.setStyleSheet(f"color: {GREEN}; font-size: 14px; font-weight: 600;")
                self.compare_label.setText(f"↓ Your price is {abs(diff_pct):.1f}% below market — good deal for buyer")
            elif diff_pct > 2:
                self.compare_label.setStyleSheet(f"color: {ORANGE}; font-size: 14px; font-weight: 600;")
                self.compare_label.setText(f"↑ Your price is {diff_pct:.1f}% above market — may sell slower")
            else:
                self.compare_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 13px;")
                self.compare_label.setText("≈ Your price matches market")
        except ValueError:
            self.compare_label.setText("")


# ── Tab 3: Analytics ──────────────────────────────────────────────────────────
class TabAnalytics(QWidget):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self._worker   = None
        self._build_ui()
        self._load_meta()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        # metric cards row
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(14)

        self.metric_widgets = {}
        for key, lbl_text, color in [
            ("mae",        "MAE",           ACCENT),
            ("rmse",       "RMSE",          ORANGE),
            ("r2",         "R²",            GREEN),
            ("n_samples",  "Training size", "#a89bff"),
        ]:
            card = card_frame()
            cl = QVBoxLayout(card)
            cl.setContentsMargins(16, 14, 16, 14)
            cl.setSpacing(4)
            cl.addWidget(label(lbl_text, color=TEXT_FAINT, size=11, bold=True))
            val_lbl = QLabel("—")
            val_lbl.setStyleSheet(f"color: {color}; font-size: 26px; font-weight: 700;")
            cl.addWidget(val_lbl)
            self.metric_widgets[key] = val_lbl
            metrics_row.addWidget(card)

        root.addLayout(metrics_row)

        # trained_at
        self.trained_label = label("Last trained: —", color=TEXT_DIM, size=12)
        root.addWidget(self.trained_label)

        # feature importance chart
        fi_card = card_frame()
        fi_layout = QVBoxLayout(fi_card)
        fi_layout.setContentsMargins(14, 12, 14, 12)
        fi_layout.addWidget(label("Feature Importance", bold=True))
        self.fi_canvas = MplCanvas(height=4)
        fi_layout.addWidget(self.fi_canvas)
        root.addWidget(fi_card, stretch=1)

        # retrain
        retrain_card = card_frame()
        rl = QHBoxLayout(retrain_card)
        rl.setContentsMargins(16, 14, 16, 14)
        rl.setSpacing(16)

        left_col = QVBoxLayout()
        left_col.addWidget(label("Retrain model", bold=True))
        left_col.addWidget(label("Parse fresh data & retrain XGBoost", color=TEXT_FAINT, size=11))
        rl.addLayout(left_col)

        progress_col = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.status_label = label("Ready", color=TEXT_DIM, size=11)
        progress_col.addWidget(self.progress_bar)
        progress_col.addWidget(self.status_label)
        rl.addLayout(progress_col, stretch=1)

        self.btn_retrain = QPushButton("Retrain")
        self.btn_retrain.clicked.connect(self._start_retrain)
        rl.addWidget(self.btn_retrain)

        root.addWidget(retrain_card)

    def _load_meta(self):
        if not self.predictor:
            return
        meta = self.predictor.get_meta()
        if "error" in meta:
            return
        metrics = meta.get("metrics", {})
        mae  = metrics.get("mae")
        rmse = metrics.get("rmse")
        r2   = metrics.get("r2")
        n    = meta.get("n_samples")

        if mae  is not None: self.metric_widgets["mae"].setText(f"{mae:.0f} BYN")
        if rmse is not None: self.metric_widgets["rmse"].setText(f"{rmse:.0f} BYN")
        if r2   is not None: self.metric_widgets["r2"].setText(f"{r2:.3f}")
        if n    is not None: self.metric_widgets["n_samples"].setText(str(n))

        trained_at = meta.get("trained_at", "")
        if trained_at:
            try:
                dt = datetime.fromisoformat(trained_at)
                self.trained_label.setText(f"Last trained: {dt.strftime('%d.%m.%Y %H:%M')}")
            except Exception:
                self.trained_label.setText(f"Last trained: {trained_at}")

        self._draw_feature_importance()

    def _draw_feature_importance(self):
        if not self.predictor or self.predictor.model is None:
            return
        try:
            model = self.predictor.model
            reg = model.named_steps.get("regressor")
            if reg is None or not hasattr(reg, "feature_importances_"):
                return

            importances = reg.feature_importances_
            extractor = model.named_steps.get("extractor")
            if extractor and hasattr(extractor, "feature_names_out_"):
                names = list(extractor.feature_names_out_)
            else:
                names = [f"f{i}" for i in range(len(importances))]

            pairs = sorted(zip(names, importances), key=lambda x: x[1])[-15:]
            names_, vals_ = zip(*pairs)

            ax = self.fi_canvas.ax
            ax.clear()
            bars = ax.barh(names_, vals_, color=ACCENT, alpha=0.85)
            ax.set_facecolor(BG)
            ax.tick_params(colors=TEXT_FAINT, labelsize=9)
            ax.set_xlabel("Importance", color=TEXT_DIM, fontsize=9)
            for spine in ax.spines.values():
                spine.set_color(BORDER)
            self.fi_canvas.fig.tight_layout(pad=0.8)
            self.fi_canvas.draw()
        except Exception as e:
            print(f"Feature importance error: {e}")

    def _start_retrain(self):
        self.btn_retrain.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting…")
        self.status_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")

        self._worker = RetrainWorker()
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def _on_finished(self, meta):
        self.progress_bar.setValue(100)
        self.status_label.setStyleSheet(f"color: {GREEN}; font-size: 11px; font-weight: 600;")
        self.status_label.setText("✓ Model updated successfully")
        self.btn_retrain.setEnabled(True)
        if self.predictor:
            self.predictor.model = __import__("joblib").load(self.predictor.model_path)
        self._load_meta()

    def _on_error(self, err):
        self.status_label.setStyleSheet(f"color: {RED}; font-size: 11px;")
        self.status_label.setText(f"Error: {err}")
        self.btn_retrain.setEnabled(True)


# ── Top bar ───────────────────────────────────────────────────────────────────
class TopBar(QFrame):
    def __init__(self, tab_names, meta):
        super().__init__()
        self.setObjectName("topbar")
        self.setFixedHeight(52)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 0, 18, 0)

        self.tab_label = label("Market Analysis", bold=True, size=15)
        layout.addWidget(self.tab_label)
        layout.addStretch()

        trained = meta.get("trained_at", "")
        mae     = meta.get("metrics", {}).get("mae")
        trained_str = ""
        if trained:
            try:
                trained_str = datetime.fromisoformat(trained).strftime("%d.%m.%Y")
            except Exception:
                trained_str = trained

        status_parts = []
        if trained_str:
            status_parts.append(f"Trained: {trained_str}")
        if mae:
            status_parts.append(f"MAE: {mae:.0f} BYN")
        status_parts.append("XGBoost")

        status = label("  ·  ".join(status_parts), color=TEXT_DIM, size=11)
        status.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 11px; background: #1a1a26; "
            f"border: 1px solid {BORDER}; border-radius: 7px; padding: 5px 12px;"
        )
        layout.addWidget(status)

        dot = QLabel("●")
        dot.setStyleSheet(f"color: {GREEN}; font-size: 10px;")
        layout.addWidget(dot)

    def set_tab(self, name):
        self.tab_label.setText(name)


# ── Main window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic Pricing Optimizer")
        self.resize(1300, 860)

        self.predictor = self._load_predictor()
        self.analyzer  = self._load_analyzer()

        meta = self.predictor.get_meta() if self.predictor else {}

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        TAB_NAMES = ["Market Analysis", "Price Prediction", "Analytics"]

        self.topbar = TopBar(TAB_NAMES, meta)
        layout.addWidget(self.topbar)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(
            lambda i: self.topbar.set_tab(TAB_NAMES[i])
        )
        layout.addWidget(self.tabs)

        self.tabs.addTab(TabMarket(self.analyzer, self.predictor),   "📊  Market Analysis")
        self.tabs.addTab(TabPredict(self.analyzer, self.predictor),  "🎯  Price Prediction")
        self.tabs.addTab(TabAnalytics(self.predictor),               "📈  Analytics")

    def _load_predictor(self):
        try:
            base = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, os.path.join(base, ".."))
            from predictor import Predictor
            model_path = os.path.join(base, "..", "models", "xgb_pipeline.pkl")
            meta_path  = os.path.join(base, "..", "models", "model_meta.json")
            p = Predictor(model_path=model_path, meta_path=meta_path)
            return p
        except Exception as e:
            print(f"Predictor not loaded: {e}")  # уже есть, проверь вывод в консоли
            return None

    def _load_analyzer(self):
        try:
            base = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, os.path.join(base, ".."))
            from market_analyzer import MarketAnalyzer
            data_path = os.path.join(base, "..", "data", "raw")
            return MarketAnalyzer(data_path=data_path)
        except Exception as e:
            print(f"Analyzer not loaded: {e}")
            return None


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())