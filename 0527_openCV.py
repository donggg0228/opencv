import os
import cv2
from deepface import DeepFace

# 載入 Haar Cascade 人臉模型
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 資料夾設定
input_dir = "data"
output_dir = "output"

# 建立 output 資料夾
os.makedirs(output_dir, exist_ok=True)

# 支援的圖片格式
image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

# 逐一讀取 data 資料夾中的圖片
for filename in os.listdir(input_dir):

    if not filename.lower().endswith(image_extensions):
        continue

    image_path = os.path.join(input_dir, filename)

    print(f"Processing: {filename}")

    # 讀取圖片
    img = cv2.imread(image_path)

    if img is None:
        print(f"Cannot read image: {filename}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 偵測人臉
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # 分析每張臉
    for (x, y, w, h) in faces:

        face_img = img[y:y+h, x:x+w]

        try:
            result = DeepFace.analyze(
                img_path=face_img,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="skip"
            )

            emotion = result[0]["dominant_emotion"]

            # 畫框與文字
            cv2.rectangle(
                img,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                img,
                emotion,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        except Exception as e:
            print(f"Error analyzing face in {filename}: {e}")

    # 儲存結果
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, img)

    print(f"Saved: {output_path}")

print("All images processed.")