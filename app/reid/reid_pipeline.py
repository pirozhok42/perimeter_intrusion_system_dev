from pathlib import Path
import json
import cv2
from tqdm import tqdm

from app.config import AppConfig
from app.detection.yolo_detector import YoloDetector
from app.tracking.bytetrack_tracker import ByteTrackTracker
from app.reid.osnet_extractor import OsnetExtractor
from app.reid.global_id_manager import GlobalIdManager
from app.video.video_io import make_video_writer
from app.gui.services.preview_service import save_frame


class ReIdPipeline:
    def __init__(self, config: AppConfig, global_id_manager=None):
        self.config = config
        reid_cfg = config.raw.get("reid", {})

        self.detector = YoloDetector(
            config.model_weights,
            config.confidence_threshold,
            config.iou_threshold,
            config.enabled_class_ids
        )

        self.extractor = OsnetExtractor(
            model_name=reid_cfg.get("model_name", "mobilenet_v3_large"),
            device=reid_cfg.get("device", "cpu"),
            weights_path=reid_cfg.get("weights", "mobilenet_v3_large.pth")
        )

        self.global_ids = global_id_manager or GlobalIdManager(
            similarity_threshold=float(reid_cfg.get("similarity_threshold", 0.72))
        )

        # local ByteTrack id -> stable global id inside this camera/video
        self.local_to_global = {}

        # do ReID not every frame for same local track, it reduces ID flickering
        self.reid_update_every_n_frames = int(reid_cfg.get("reid_update_every_n_frames", 10))

    def process_video(self, video_path):
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_video_dir = Path("processed/videos")
        output_log_dir = Path("processed/logs")
        output_video_dir.mkdir(parents=True, exist_ok=True)
        output_log_dir.mkdir(parents=True, exist_ok=True)

        output_video = output_video_dir / f"{video_path.stem}_reid_annotated.mp4"
        output_json = output_log_dir / f"{video_path.stem}_reid_events.json"

        writer = make_video_writer(output_video, fps, width, height)
        tracker = ByteTrackTracker()

        events = []
        frame_idx = 0
        saved_reid_event_preview = False

        with tqdm(total=total_frames if total_frames > 0 else None, desc=f"ReID {video_path.name}", disable=False) as pbar:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                detections = self.detector.detect(frame)
                detections = tracker.update(detections)

                frame_has_person = False

                for i in range(len(detections)):
                    xyxy = detections.xyxy[i].astype(int)
                    x1, y1, x2, y2 = xyxy

                    local_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
                    confidence = float(detections.confidence[i]) if detections.confidence is not None else 0.0

                    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    if crop.size == 0:
                        continue

                    frame_has_person = True

                    # First time seeing this local track: match against shared global gallery.
                    if local_id not in self.local_to_global:
                        embedding = self.extractor.extract(crop)
                        if embedding is None:
                            continue

                        global_id, score, is_new = self.global_ids.match(embedding)
                        self.local_to_global[local_id] = global_id

                    else:
                        global_id = self.local_to_global[local_id]
                        score = 1.0
                        is_new = False

                        # Occasionally update global gallery with fresher crop.
                        if frame_idx % self.reid_update_every_n_frames == 0:
                            embedding = self.extractor.extract(crop)
                            self.global_ids.update_existing(global_id, embedding)

                    color = (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"GID {global_id} LID {local_id}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2
                    )

                    events.append({
                        "frame": frame_idx,
                        "timestamp_sec": round(frame_idx / fps, 3),
                        "camera_id": self.config.camera_id,
                        "local_id": int(local_id),
                        "object_id": int(global_id),
                        "event_type": "REID_PERSON_DETECTED",
                        "similarity_score": round(float(score), 4),
                        "is_new_global_id": bool(is_new),
                        "confidence_score": round(confidence, 4),
                    })

                writer.write(frame)

                if frame_has_person and not saved_reid_event_preview:
                    save_frame(self.config.camera_id, frame, event=True)
                    saved_reid_event_preview = True
                elif not frame_has_person and frame_idx % 15 == 0:
                    save_frame(self.config.camera_id, frame, event=False)

                frame_idx += 1
                pbar.update(1)

        cap.release()
        writer.release()

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)

        return {
            "input_video": str(video_path),
            "output_video": str(output_video),
            "events_json": str(output_json),
            "events_count": len(events),
        }
