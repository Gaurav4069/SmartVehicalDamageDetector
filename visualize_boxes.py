import requests
import cv2
import random

API_KEY = "gL5QCD8pKajC4v5pr3X3"
MODEL_ID = "car-damage-detection-t0g92/3"
API_URL = f"https://detect.roboflow.com/{MODEL_ID}"


class DamageExtractorAPI:

    def __init__(self):
        pass

    # ----------------------------------------------------
    # API Request to Roboflow
    # ----------------------------------------------------
    def infer_api(self, image_path):
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = requests.post(
            API_URL,
            params={"api_key": API_KEY},
            files={"file": image_bytes}
        )
        return response.json()

    # ----------------------------------------------------
    # MAIN EXTRACTION LOGIC
    # ----------------------------------------------------
    def extract(self, image_path):
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        img_area = h * w if h and w else 1

        result = self.infer_api(image_path)
        preds = result.get("predictions", [])

        damaged_parts = {}
        total_area = 0

        for pred in preds:
            part = pred["class"]     # e.g. "Door", "Bumper", etc.
            width = pred["width"]
            height = pred["height"]

            # bounding box area
            box_area = width * height
            total_area += box_area

            # count parts
            if part not in damaged_parts:
                damaged_parts[part] = 0
            damaged_parts[part] += 1

        # compute damage area ratio
        damage_ratio = total_area / img_area

        # SEVERITY BASED ON BBOX AREA
        if len(preds) == 0:
            severity = "no_damage"
        elif damage_ratio < 0.02:
            severity = "minor"
        elif damage_ratio < 0.08:
            severity = "moderate"
        else:
            severity = "severe"

        return {
            "severity": severity,
            "damage_ratio": damage_ratio,
            "damaged_parts": damaged_parts,
            "num_damaged_parts": len(damaged_parts),
            "raw_predictions": preds
        }

    # ----------------------------------------------------
    # VISUALIZATION FUNCTION
    # ----------------------------------------------------
    def visualize(self, image_path, predictions, save_path="output.jpg"):
        img = cv2.imread(image_path)

        if img is None:
            print("❌ Could not load image.")
            return

        for pred in predictions:
            x = pred["x"]
            y = pred["y"]
            w = pred["width"]
            h = pred["height"]
            cls = pred["class"]
            conf = pred["confidence"]

            # YOLO = center format → convert to top-left format
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # random color for each class
            color = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )

            # draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            # label
            label = f"{cls} ({conf:.2f})"
            cv2.putText(
                img, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2
            )

        # save output
        cv2.imwrite(save_path, img)

        # show output in window
        cv2.imshow("Detections", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("✔ Saved visualization:", save_path)
