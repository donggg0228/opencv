import os
import cv2
from deepface import DeepFace

# =====================================================================
# 步驟 1：安全的載入 Haar Cascade 人臉模型
# =====================================================================
xml_name = "haarcascade_frontalface_default.xml"
face_cascade = None

# 嘗試路徑 A：檢查目前程式的同目錄下有沒有該 XML 檔案
if os.path.exists(xml_name):
    face_cascade = cv2.CascadeClassifier(xml_name)
    print("成功從【目前目錄】載入人臉特徵模型。")

# 嘗試路徑 B：如果同目錄沒有，嘗試使用 OpenCV 預設內建路徑（並用 os.path.join 確保斜線正確）
if face_cascade is None or face_cascade.empty():
    default_xml_path = os.path.abspath(
        os.path.join(cv2.data.haarcascades, xml_name)
    )
    face_cascade = cv2.CascadeClassifier(default_xml_path)
    if not face_cascade.empty():
        print(f"成功從【OpenCV 系統目錄】載入模型：{default_xml_path}")

# 如果以上方法都失敗，給出明確錯誤提示並中斷，防止後續 detectMultiScale 報錯
if face_cascade is None or face_cascade.empty():
    print("\n" + "=" * 60)
    print(f"【錯誤】找不到人臉模型檔案：'{xml_name}'")
    print("請至以下網址下載檔案，並將它放至與此 Python 程式【相同的資料夾】內：")
    print(
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    )
    print("=" * 60 + "\n")
    exit()  # 結束程式

# =====================================================================
# 步驟 2：資料夾設定與初始化
# =====================================================================
input_base_dir = "data"  # 裡面含有 Sad, Angry, Happy 三個子資料夾
output_dir = "face_data_ok"

# 建立輸出資料夾
os.makedirs(output_dir, exist_ok=True)

# 支援的圖片格式
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

print("\n開始處理圖片...")
print("-" * 50)

# =====================================================================
# 步驟 3：走訪資料夾並進行臉部辨識
# =====================================================================
# 使用 os.walk 自動深入 data 底下的所有子資料夾
for root, dirs, files in os.walk(input_base_dir):
    for filename in files:
        if not filename.lower().endswith(image_extensions):
            continue

        # 取得圖片在子資料夾中的完整路徑
        image_path = os.path.join(root, filename)
        print(f"正在處理: {image_path}")

        # 讀取圖片
        img = cv2.imread(image_path)
        if img is None:
            print(f" └ [失敗] 無法讀取圖片: {image_path}")
            continue

        # 轉為灰階（供 Haar Cascade 偵測人臉使用）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 偵測人臉
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # 針對這張圖片中偵測到的每張臉進行分析
        face_count = 0
        for x, y, w, h in faces:
            # 裁切出人臉區域
            face_img = img[y : y + h, x : x + w]

            # 確保裁切區塊大小正確，避免空矩陣丟入 DeepFace 報錯
            if face_img.size == 0:
                continue

            try:
                # 利用 DeepFace 做情緒辨識
                # detector_backend='skip' 表示直接分析傳入的 face_img，不用讓 DeepFace 再偵測一次臉
                result = DeepFace.analyze(
                    img_path=face_img,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="skip",
                )

                # 取得分數最高的主導情緒
                emotion = result[0]["dominant_emotion"]
                face_count += 1

                # 在原圖上畫上綠色框 (BGR: 0, 255, 0)，線條粗細為 2
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 在框的上方 (-10 像素位置) 寫上辨識出的英文情緒名稱
                cv2.putText(
                    img,
                    emotion,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,  # 字體大小
                    (0, 255, 0),  # 字體顏色
                    2,  # 字體線條粗細
                )
                print(f" └ 偵測到第 {face_count} 張臉 -> 表情: {emotion}")

            except Exception as e:
                print(f" └ [錯誤] DeepFace 分析人臉時發生問題: {e}")

        # 為了避免不同子資料夾（如 Sad, Happy）有相同的檔名（如 001.jpg）在輸出時被覆蓋
        # 我們將當前的子資料夾名稱組合進新的檔名中
        sub_folder_name = os.path.basename(root)
        if sub_folder_name and sub_folder_name != input_base_dir:
            output_filename = f"{sub_folder_name}_{filename}"
        else:
            output_filename = filename

        # 儲存結果圖片到 face_data_ok 資料夾中
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, img)
        print(f" └ [已儲存] -> {output_path}")

print("-" * 50)
print(f"所有圖片處理完畢！請至 '{output_dir}' 資料夾查看標記後的結果。")