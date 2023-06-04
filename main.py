import time
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk, ImageFont, ImageDraw
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from playsound import playsound
from gtts import gTTS
import arabic_reshaper
from PIL import Image
import tensorflow as tf
import os
import warnings
from bidi.algorithm import get_display
import onnxruntime as rt
from PIL import Image, ImageDraw, ImageColor, ImageFont
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
warnings.filterwarnings('ignore')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
cwd = os.getcwd()

WHITE_COLOR = (245, 242, 226)
RED_COLOR = (25, 35, 240)
HEIGHT = 600

############## Action Recognition Labels #################
df = pd.read_excel(r'KARSL-502_Labels.xlsx')
print(df.iloc[502-1]['Sign-English'])
actions = []
for i in range(190):
    actions.append(df.iloc[i]['Sign-Arabic'])

############## Object Detection Labels ##############
classes = {
    0: 'حرف العين',
    1: 'ال التعريف',
    2: 'حرف الألف',
    3: 'حرف الباء',
    4: 'حرف الدال',
    5: 'حرف الظاء',
    6: 'حرف الضاد',
    7: 'حرف الفاء',
    8: 'حرف القاف',
    9: 'حرف الغين',
    10: 'حرف الهاء',
    11: 'حرف الحاء',
    12: 'حرف الجيم',
    13: 'حرف الكاف',
    14: 'حرف الخاء',
    15: 'لا',
    16: 'حرف اللام',
    17: 'حرف الميم',
    18: 'حرف النون',
    19: 'حرف الراء',
    20: 'حرف الصاد',
    21: 'حرف السين',
    22: 'حرف الشين',
    23: 'حرف الطاء',
    24: 'حرف التاء',
    25: 'حرف الثاء',
    26: 'حرف الذال',
    27: 'تاء مربوطة',
    28: 'حرف الواو',
    29: 'حرف الياء',
    30: 'همزة متوسطة',
    31: 'حرف الزاي'
}
classes2 = {
    0: 'ain',
    1: 'al',
    2: 'aleff',
    3: 'bb',
    4: 'dal',
    5: 'dha',
    6: 'dhad',
    7: 'fa',
    8: 'gaaf',
    9: 'ghain',
    10: 'ha',
    11: 'haa',
    12: 'jeem',
    13: 'kaaf',
    14: 'khaa',
    15: 'la',
    16: 'laam',
    17: 'meem',
    18: 'nun',
    19: 'ra',
    20: 'saad',
    21: 'seen',
    22: 'sheen',
    23: 'ta',
    24: 'taa',
    25: 'thaa',
    26: 'thal',
    27: 'toot',
    28: 'waw',
    29: 'ya',
    30: 'yaa',
    31: 'zay'
}

############## Object Detection Model ##############
OD_MODEL = "model.onnx"
font = ImageFont.truetype(r"arial.ttf", 16)
OD_sess = rt.InferenceSession(OD_MODEL)
OD_outputs = ["detection_boxes", "detection_classes",
              "detection_scores", "num_detections"]

############## Action Recognition Model ##############
AR_MODEL = "gesture.onnx"
AR_sess = rt.InferenceSession(AR_MODEL)
AR_INPUT = ['conv2d_input']
AR_output = ['dense_1']


############## Draw bounding box and detected letter ##############
def draw_detection(img, b, c, s):
    if s < 0.05:
        return
    width, height = img.im.size
    top = b[0] * height
    left = b[1] * width
    bottom = b[2] * height
    right = b[3] * width
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(height, np.floor(bottom + 0.5).astype('int32'))
    right = min(width, np.floor(right + 0.5).astype('int32'))
    label = classes[c.astype('int32') - 1]
    reshaped = arabic_reshaper.reshape(label)
    label = get_display(reshaped)
    text_origin = tuple(np.array([left + 1, top + 1]))
    color = ImageColor.getrgb("red")
    thickness = 0
    img.rectangle([left + thickness, top + thickness, right -
                   thickness, bottom - thickness], outline=color)
    img.text(text_origin, label, fill=color, font=font)


############## Update Camera Frames ##############
def update(frame: np.ndarray, sign_detected: str, is_recording: bool):
    WIDTH = int(HEIGHT * len(frame[0]) / len(frame))
    # Resize frame
    frame = cv2.resize(frame, (WIDTH, HEIGHT),
                       interpolation=cv2.INTER_AREA)
    # Flip the image vertically for mirror effect
    frame = cv2.flip(frame, 1)
    # Write result if there is
    frame = draw_text(frame, sign_detected)
    # Chose circle color
    color = WHITE_COLOR
    if is_recording:
        color = RED_COLOR
    # Update the frame
    cv2.circle(frame, (30, 30), 20, color, -1)
    cv2.imshow("OpenCV Feed", frame)


############## Draw detected text ##############
def draw_text(frame, sign_detected, font=cv2.FONT_HERSHEY_COMPLEX, font_size=1, font_thickness=2, offset=int(HEIGHT * 0.02), bg_color=(245, 242, 176, 0.85),):

    window_w = int(HEIGHT * len(frame[0]) / len(frame))

    (text_w, text_h), _ = cv2.getTextSize(
        sign_detected, font, font_size, font_thickness
    )

    text_x, text_y = int((window_w - text_w) / 2), HEIGHT - text_h - offset

    cv2.rectangle(frame, (0, text_y - offset),
                  (window_w, HEIGHT), bg_color, -1)
    fontpath = "arial.ttf"
    font = ImageFont.truetype(fontpath, 32)
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((text_x, text_y - 5), sign_detected, font=font)
    frame = np.array(img_pil)
    return frame


############## Detect and Predict the letter in uploaded image ##############


def classify():
    try:
        img = Image.open('itbp.png')
    except:
        return
    img_data = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
    img_data = np.expand_dims(img_data.astype(np.uint8), axis=0)
    OD_sess = rt.InferenceSession(OD_MODEL)
    result = OD_sess.run(OD_outputs, {"input_tensor": img_data})
    mask = ImageDraw.Draw(img)
    detection_boxes, detection_classes, detection_scores, num_detections = result
    draw_detection(
        mask, detection_boxes[0][0], detection_classes[0][0], detection_scores[0][0])
    img.save('output.png')
    setDisplay("output.png")
    messageLabel.config(text=classes[detection_classes[0][0] - 1])

############## Upload Image ##############


def upload_image():
    global img
    f_types = [('Jpg Files', '*.jpg *.jpeg *.png')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img = Image.open(filename)
    img.convert('RGB')
    img.save("itbp.png")
    setDisplay("itbp.png")


############## Detect Keypoints for each frame ##############


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

############## Extract Keypoints from resulted Mediapipe detection for each frame ##############


def extract_keypoints(results):
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
    ) if results.face_landmarks else np.zeros(1404)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


############## Live feed detection ##############
def snapshot():
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    d = 1
    is_recording = True
    sequence = []
    sign_detected = ''
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)

            if d == -1:
                d = d * (-1)
                continue
            d = d * (-1)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            print(len(sequence))
            if (len(sequence) == 30):
                is_recording = False
                data = np.array(sequence, dtype=np.float32)
                data = np.expand_dims(data, axis=-1)
                data = np.expand_dims(data, axis=0)
                result = AR_sess.run(
                    AR_output, {AR_INPUT[0]: data})
                sign_detected = actions[np.argmax(result)]
                if isinstance(sign_detected, int):
                    sign_detected = str(sign_detected)
                else:
                    sign_detected = arabic_reshaper.reshape(sign_detected)
                    sign_detected = get_display(sign_detected)
                update(frame, sign_detected, is_recording)
                cv2.waitKey(1500)
                is_recording = True
                sequence = []

            update(frame, sign_detected, is_recording)
            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord("z"):
                sequence = []
            elif pressedKey == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


############## Display Uploaded Image ##############


def setDisplay(img1):
    global img
    img = Image.open(img1)
    resized = img.resize((500, 500), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(resized)
    picDisplay.configure(image=img)
    picDisplay.grid(rowspan=40, column=0)


############## Play Detected Letter ##############


def play():
    if os.path.exists("audio.mp3"):
        os.remove("audio.mp3")
    print(messageLabel.cget("text"))
    audio = gTTS(messageLabel.cget("text"), lang='ar')
    audio.save("audio.mp3")
    playsound("audio.mp3")


############## GUI ##############
root = Tk()
root.geometry("800x590")
root.title("Team Elite")
f1 = Frame(root)
picDisplay = Button(root, text="image", padx=230, pady=230,
                    borderwidth=3, relief="groove")
uploadButton = Button(root, text="Upload Image", width=30,
                      command=lambda: upload_image())
captureButton = Button(root, text="Capture Image",
                       width=30, command=lambda: snapshot())
detectButton = Button(root, text="Detect Gesture",
                      width=30, command=lambda: classify())
speakButton = Button(root, text="Speak Text", width=30, command=lambda: play())
exitButton = Button(root, text="Exit", width=30, command=root.destroy)
messageLabel = Label(root, text="Status: idle")
picDisplay.grid(rowspan=40, column=0, padx=10, pady=10)
uploadButton.grid(row=5, column=1, padx=10)
captureButton.grid(row=6, column=1, padx=10)
detectButton.grid(row=7, column=1, padx=10)
speakButton.grid(row=8, column=1, padx=10)
exitButton.grid(row=41, column=1, padx=10)
messageLabel.grid(row=41, column=0, padx=10, pady=10)
root.mainloop()
