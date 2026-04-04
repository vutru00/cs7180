import math

import cv2
import numpy as np
from PIL import Image


class MiVOLOClassifier:
    """Wraps a MiVOLO Predictor into a clean scoring interface."""

    def __init__(self, predictor):
        self.predictor = predictor

    def predict_single(self, pil_image):
        """Run age/gender prediction on a single PIL image.

        Returns dict with keys {age, gender, gender_score, class_name}
        or None if no human is detected.
        """
        bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        detected_objects, _ = self.predictor.recognize(bgr)

        rows = []
        names = detected_objects.yolo_results.names
        boxes = detected_objects.yolo_results.boxes

        for idx, (det, age, gender, gender_score) in enumerate(
            zip(boxes, detected_objects.ages, detected_objects.genders, detected_objects.gender_scores)
        ):
            if age is None or gender is None:
                continue

            cls_name = names[int(det.cls)]
            conf = float(det.conf)
            x1, y1, x2, y2 = det.xyxy.squeeze().detach().cpu().numpy().tolist()
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

            rows.append({
                "class_name": cls_name,
                "det_conf": conf,
                "age": float(age),
                "gender": str(gender),
                "gender_score": float(gender_score) if gender_score is not None else float("nan"),
                "bbox_area": area,
            })

        if not rows:
            return None

        # Prefer face detections over person detections
        face_rows = [r for r in rows if r["class_name"] == "face"]
        person_rows = [r for r in rows if r["class_name"] == "person"]
        candidates = face_rows if face_rows else person_rows

        # Highest confidence, break ties by larger bounding box area
        candidates.sort(key=lambda r: (r["det_conf"], r["bbox_area"]), reverse=True)
        best = candidates[0]

        return {
            "age": best["age"],
            "gender": best["gender"],
            "gender_score": best["gender_score"],
            "class_name": best["class_name"],
        }

    def extract_bias_score(self, images, dim="gender"):
        """Score a batch of PIL images.

        Args:
            images: list of PIL Images
            dim: "gender" returns male probability [0,1], "age" returns age float.

        Returns list of floats (NaN for non-detections).
        """
        scores = []
        for img in images:
            result = self.predict_single(img)
            if result is None:
                scores.append(float("nan"))
                continue

            if dim == "gender":
                # Convert to male probability: 1.0 = confident male, 0.0 = confident female
                raw = result["gender_score"]
                if result["gender"] == "male":
                    scores.append(raw)
                else:
                    scores.append(1.0 - raw)
            elif dim == "age":
                scores.append(result["age"])
            else:
                raise ValueError(f"Unknown bias dimension: {dim}")

        return scores
