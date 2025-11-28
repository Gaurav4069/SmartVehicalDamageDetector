from damage_extractor_api import DamageExtractorAPI
from tkinter import Tk, filedialog

# hide empty tkinter window
Tk().withdraw()

# browse file
file_path = filedialog.askopenfilename(
    title="Select a car image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)

if not file_path:
    print("❌ No file selected.")
else:
    det = DamageExtractorAPI()
    result = det.extract(file_path)

    print("\n📌 Selected File:", file_path)
    print("📌 Result:", result)

    # 🔥 visualize bounding boxes (NEW)
    det.visualize(file_path, result["raw_predictions"], "output.jpg")
