import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import paddlehub as hub
import cv2


"""
" @Author WEIHUANG
" 
" @Data 21/4/2021
"
" @By XBMU
"
" @E-mail 18407545990@163.com
"""


mask_detector = hub.Module(name="pyramidbox_lite_server_mask")

cap = cv2.VideoCapture(0)
Open = cap.isOpened()
if not Open:
    print("打开摄像头")
while Open:
    ret, frame = cap.read()
    if not ret:
        print("Video is Over")
        break
    result = mask_detector.face_detection(images=[frame])
    if result:
        obj_dict = result[0]
        data_list = obj_dict["data"]
        for obj in data_list:
            label = obj["label"]
            conf = obj["confidence"]
            if conf < 0.95 or label == "MASK":
                xmin, ymin, xmax, ymax = obj["left"], obj["top"], obj["right"], obj["bottom"]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2, 1)
                cv2.putText(frame, "Has Mask", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,8,0)
                continue
            xmin, ymin, xmax, ymax = obj["left"], obj["top"], obj["right"], obj["bottom"]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2, 1)
            cv2.putText(frame, "No Mask", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,8,0)
    cv2.namedWindow("frame", 0)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()