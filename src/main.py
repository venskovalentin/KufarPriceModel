import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QTableWidget,
                             QTableWidgetItem, QHeaderView, QMessageBox)
from PyQt6.QtCore import Qt

from market_analyzer import MarketAnalyzer
from predictor import Predictor


class KufarApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализатор рынка ноутбуков (Kufar)")
        self.resize(1000, 600)

        self.analyzer = None
        self.predictor = None

        # Вычисляем абсолютный путь к корню проекта (на уровень выше, чем папка src)
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.init_data()
        self.init_ui()

    def init_data(self):
        """Инициализируем анализатор и предиктор, используя абсолютные пути"""
        data_path = os.path.join(self.base_dir, 'data', 'raw')
        model_path = os.path.join(self.base_dir, 'models', 'xgb_pipeline.pkl')
        meta_path = os.path.join(self.base_dir, 'models', 'model_meta.json')

        try:
            self.analyzer = MarketAnalyzer(data_path=data_path)
            self.predictor = Predictor(model_path=model_path, meta_path=meta_path)
        except FileNotFoundError as e:
            self.show_error(f"Ошибка загрузки данных: {e}")
        except Exception as e:
            self.show_error(f"Неизвестная ошибка: {e}")

    def init_ui(self):
        """Построение интерфейса"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Верхняя панель (Статистика и Кнопки) ---
        top_layout = QHBoxLayout()

        self.stats_label = QLabel("Нажмите 'Обновить статистику' для загрузки данных.")
        self.stats_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        top_layout.addWidget(self.stats_label)

        button_layout = QVBoxLayout()

        self.btn_stats = QPushButton("Обновить статистику")
        self.btn_stats.clicked.connect(self.update_stats)

        self.btn_deals = QPushButton("Найти недооцененные (Выгода > 15%)")
        self.btn_deals.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_deals.clicked.connect(self.show_underpriced)

        button_layout.addWidget(self.btn_stats)
        button_layout.addWidget(self.btn_deals)
        top_layout.addLayout(button_layout)

        main_layout.addLayout(top_layout)

        # --- Нижняя панель (Таблица) ---
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "ID Объявления", "Описание", "Текущая цена (BYN)",
            "Прогноз модели (BYN)", "Выгода (BYN)", "Скидка (%)"
        ])

        # ИСПРАВЛЕНИЕ 1: Добавлено .ResizeMode. для PyQt6
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        main_layout.addWidget(self.table)

        # Загружаем базовую статистику при старте
        if self.analyzer:
            self.update_stats()

    def update_stats(self):
        if not self.analyzer:
            return

        stats = self.analyzer.get_market_stats()
        text = (f"<b>Статистика рынка:</b><br>"
                f"Количество объявлений: {stats['count']}<br>"
                f"Средняя цена: {stats['mean']} BYN<br>"
                f"Медианная цена: {stats['median']} BYN<br>"
                f"Диапазон: {stats['min']} - {stats['max']} BYN")
        self.stats_label.setText(text)

    def show_underpriced(self):
        if not self.analyzer or not self.predictor:
            self.show_error("Анализатор или модель не загружены.")
            return

        try:
            # Ищем ноутбуки, которые стоят меньше 85% от предсказанной цены
            deals_df = self.analyzer.find_underpriced(self.predictor, threshold=0.85)

            if deals_df.empty:
                QMessageBox.information(self, "Результат", "Не найдено сильно недооцененных ноутбуков.")
                return

            self.populate_table(deals_df)

        except Exception as e:
            self.show_error(f"Ошибка при поиске выгодных предложений: {e}")

    def populate_table(self, df):
        self.table.setRowCount(0)  # Очищаем старые данные
        self.table.setRowCount(len(df))

        for row_idx, (_, row) in enumerate(df.iterrows()):
            # Безопасное получение ad_id, если он есть
            ad_id = str(row.get('ad_id', 'Н/Д'))
            subject = str(row.get('subject', 'Без названия'))
            price = f"{row['price_byn']:.2f}"
            predicted = f"{row['predicted_price']:.2f}"
            profit = f"{row['profit_byn']:.2f}"

            # Переводим discount_ratio (например, 0.70) в понятную скидку (30%)
            discount_percent = (1 - row['discount_ratio']) * 100
            discount_str = f"{discount_percent:.1f}%"

            # Заполняем ячейки
            self.table.setItem(row_idx, 0, QTableWidgetItem(ad_id))
            self.table.setItem(row_idx, 1, QTableWidgetItem(subject))

            # Цены выравниваем по правому краю
            for col, val in enumerate([price, predicted, profit, discount_str], start=2):
                item = QTableWidgetItem(val)

                # ИСПРАВЛЕНИЕ 2: Добавлено .AlignmentFlag. для PyQt6
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

                self.table.setItem(row_idx, col, item)

    def show_error(self, message):
        QMessageBox.critical(self, "Ошибка", message)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Применяем базовый стиль для красоты
    app.setStyle("Fusion")

    window = KufarApp()
    window.show()
    sys.exit(app.exec())