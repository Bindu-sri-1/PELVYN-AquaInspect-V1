import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "plant_collection"
OUT_DIR = ROOT / "plant_collection_labels"
OUT_DIR.mkdir(exist_ok=True)

images = sorted(p for p in IMG_DIR.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

if not images:
    print("Plant images dorakaledhu.")
    raise SystemExit

i = 0
while i < len(images):
    path = images[i]
    original = cv2.imread(str(path))
    if original is None:
        i += 1
        continue

    h, w = original.shape[:2]
    scale = min(1000 / w, 650 / h, 1.0)
    frame = cv2.resize(original, (int(w * scale), int(h * scale)))
    display = frame.copy()
    box = []
    drawing = False
    start = None

    def mouse(event, x, y, flags, param):
        global drawing, start, box, display
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start = (x, y)
            box = []
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            display = frame.copy()
            cv2.rectangle(display, start, (x, y), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x1, y1 = start
            box = [min(x1,x), min(y1,y), max(x1,x), max(y1,y)]
            display = frame.copy()
            cv2.rectangle(display, (box[0],box[1]), (box[2],box[3]), (0,255,0), 2)

    cv2.namedWindow("Plant Labeling", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Plant Labeling", 1100, 750)
    cv2.setMouseCallback("Plant Labeling", mouse)

    while True:
        screen = display.copy()
        cv2.putText(screen, f"{i+1}/{len(images)} - {path.name}",
                    (10,25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 2)
        cv2.putText(screen, "Drag around plant | S=Save  N=Next  P=Previous  R=Reset  Q=Quit",
                    (10,50), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,0), 1)
        cv2.imshow("Plant Labeling", screen)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("s"):
            if box and box[2] > box[0] and box[3] > box[1]:
                ih, iw = frame.shape[:2]
                x1,y1,x2,y2 = box
                cx = ((x1+x2)/2)/iw
                cy = ((y1+y2)/2)/ih
                bw = (x2-x1)/iw
                bh = (y2-y1)/ih
                (OUT_DIR / f"{path.stem}.txt").write_text(
                    f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                print("Saved:", path.name)
            else:
                print("Mundhu plant chuttu box draw cheyyi.")
        elif key == ord("r"):
            box = []
            display = frame.copy()
        elif key == ord("n"):
            i += 1
            break
        elif key == ord("p"):
            i = max(0, i-1)
            break
        elif key == ord("q"):
            cv2.destroyAllWindows()
            print("Labels saved in plant_collection_labels")
            raise SystemExit

    cv2.destroyAllWindows()

print("All images checked. Labels folder:", OUT_DIR)
