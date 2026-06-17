import json
from pathlib import Path
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QScrollArea, QGridLayout, QPushButton, QComboBox
from app.gui.services.camera_service import list_cameras
from app.gui.services.event_state_service import get_camera_status, acknowledge_summary_camera

class SummaryPage(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0,0,0,0)
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(32,32,32,32)
        title = QLabel("Сводка")
        title.setObjectName("Title")
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_holder = QWidget()
        self.grid = QGridLayout(self.grid_holder)
        self.grid.setSpacing(16)
        self.scroll.setWidget(self.grid_holder)
        self.filter_select = QComboBox()
        self.filter_select.addItems(["Все камеры", "Камеры детекции", "Камеры трекинга"])
        self.filter_select.currentTextChanged.connect(self.refresh)
        layout.addWidget(title)
        layout.addWidget(self.filter_select)
        layout.addWidget(self.scroll)
        root.addWidget(card)
        self.refresh()

    def refresh(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget: widget.deleteLater()
        mode = self.filter_select.currentText() if hasattr(self, "filter_select") else "Все камеры"
        if mode == "Камеры детекции":
            cameras = list_cameras(app_role="detection")
        elif mode == "Камеры трекинга":
            cameras = list_cameras(app_role="tracking")
        else:
            cameras = list_cameras()
        if not cameras:
            empty = QLabel("Камер пока нет.")
            empty.setObjectName("Subtitle")
            self.grid.addWidget(empty,0,0)
            return
        for idx, camera in enumerate(cameras):
            self.grid.addWidget(self._make_box(camera), idx//2, idx%2)

    def _make_box(self, camera):
        camera_name = camera["name"]
        module = 'tracking' if camera.get('app_role') == 'tracking' else 'detection'
        status = get_camera_status(camera_name, module)
        severity = status.get("severity")
        box = QFrame()
        box.setObjectName("AlarmCard" if severity=="alarm" else "WarningCard" if severity=="warning" else "CameraCard")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(18,18,18,18)
        title = QLabel(camera_name)
        title.setObjectName("SectionTitle")
        last = QLabel(f"Последнее событие: {(status.get('event_type') if status.get('severity') else 'Событий не обнаружено')}")
        last.setObjectName("Subtitle")
        last.setWordWrap(True)
        events_scroll = QScrollArea()
        events_scroll.setWidgetResizable(True)
        events_scroll.setMinimumHeight(180)
        holder = QWidget()
        ev_layout = QVBoxLayout(holder)
        events = self._load_events(status.get("log_path"))
        if not events:
            label = QLabel("Событий нет.")
            label.setObjectName("Subtitle")
            ev_layout.addWidget(label)
        else:
            for event in events:
                label = QLabel(self._format_event(event))
                label.setObjectName("Subtitle")
                label.setWordWrap(True)
                ev_layout.addWidget(label)
        ev_layout.addStretch()
        events_scroll.setWidget(holder)
        layout.addWidget(title)
        layout.addWidget(last)
        layout.addWidget(events_scroll)
        if severity:
            ok = QPushButton("Ок")
            ok.setObjectName("SecondaryButton")
            ok.clicked.connect(lambda _, c=camera_name, m=module: self._ack(c, m))
            layout.addWidget(ok)
        return box

    def _load_events(self, log_path):
        if not log_path: return []
        p = Path(log_path)
        if not p.exists(): return []
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _format_event(self, e):
        return f"• {e.get('event_type','UNKNOWN')}; время: {e.get('timestamp_sec', e.get('time_sec','?'))} сек.; кадр: {e.get('frame','?')}; объект: {e.get('object_id','?')}; класс: {e.get('class_name', e.get('class','?'))}; уверенность: {e.get('confidence_score','?')}"

    def _ack(self, camera_name, module):
        acknowledge_summary_camera(camera_name, module)
        self.refresh()