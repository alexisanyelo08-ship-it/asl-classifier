import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2
import pickle
import numpy as np
import mediapipe as mp
import subprocess
import platform

MODEL_PATH = './hand_landmarker.task'

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,
)

# Hand skeleton connections for manual drawing
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
model_classes = model.classes_

cap = cv2.VideoCapture(0)

typed_text = ""
last_char = None
stable_count = 0
STABLE_THRESHOLD = 60


def disambiguate_m_n(proba, hand):
    m_idx = list(model_classes).index('M')
    n_idx = list(model_classes).index('N')
    m_prob = proba[m_idx]
    n_prob = proba[n_idx]

    # Only override when M and N are close; otherwise trust the model
    if abs(m_prob - n_prob) > 0.25:
        return 'M' if m_prob > n_prob else 'N'

    # Ring finger MCP=13, TIP=16. y increases downward.
    # M: ring tip clearly tucked below MCP; N: ring tip at or above MCP
    ring_drop = hand[16].y - hand[13].y
    return 'M' if ring_drop > 0.02 else 'N'


_speaking_proc = None

def speak(text):
    global _speaking_proc
    if _speaking_proc:
        _speaking_proc.terminate()
    if platform.system() == 'Darwin':
        _speaking_proc = subprocess.Popen(['say', text])
    else:
        safe = text.replace("'", "''")
        _speaking_proc = subprocess.Popen(
            ['PowerShell', '-Command',
             f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe}')"]
        )


def draw_hand(frame, hand, W, H):
    pts = [(int(lm.x * W), int(lm.y * H)) for lm in hand]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)
        cv2.circle(frame, pt, 4, (0, 0, 0), 1)


with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        data_aux = []
        x_, y_ = [], []
        predicted_character = None

        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            draw_hand(frame, hand, W, H)

            x_ = [lm.x for lm in hand]
            y_ = [lm.y for lm in hand]
            for lm in hand:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            x1 = max(int(min(x_) * W) - 10, 0)
            y1 = max(int(min(y_) * H) - 10, 0)
            x2 = min(int(max(x_) * W) + 10, W)
            y2 = min(int(max(y_) * H) + 10, H)

            proba = model.predict_proba([np.asarray(data_aux)])[0]
            top_idx = int(np.argmax(proba))
            top_conf = proba[top_idx]
            predicted_character = str(model_classes[top_idx])

            if predicted_character in ('M', 'N'):
                predicted_character = disambiguate_m_n(proba, hand)

            if predicted_character == last_char:
                stable_count += 1
            else:
                last_char = predicted_character
                stable_count = 0

            if stable_count == STABLE_THRESHOLD:
                if predicted_character == 'space':
                    typed_text += ' '
                elif predicted_character == 'backspace':
                    typed_text = typed_text[:-1]
                elif predicted_character == 'enter':
                    if typed_text.strip():
                        speak(typed_text)
                    typed_text = ''
                else:
                    typed_text += predicted_character
                stable_count = 0
                last_char = None

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)

            # Current letter + confidence
            lx, ly = x1 + 5, max(y1 - 14, 30)
            label = predicted_character
            cv2.putText(frame, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 8, cv2.LINE_AA)
            cv2.putText(frame, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3, cv2.LINE_AA)

            # Progress bar
            bar_w = int((stable_count / STABLE_THRESHOLD) * (x2 - x1))
            cv2.rectangle(frame, (x1, y2 + 5), (x2, y2 + 16), (40, 40, 40), -1)
            cv2.rectangle(frame, (x1, y2 + 5), (x1 + bar_w, y2 + 16), (0, 220, 0), -1)

        # Fixed sentence bar at the top of the screen (always visible, never overlaps)
        display = typed_text[-50:] if len(typed_text) > 50 else typed_text
        cv2.rectangle(frame, (0, 0), (W, 50), (255, 255, 255), -1)
        cv2.putText(frame, display, (10, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 180), 2, cv2.LINE_AA)

        cv2.imshow('Sign Language Keyboard', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        # Keyboard Enter immediately registers whatever letter is on screen
        if key == 13 and predicted_character is not None:
            if predicted_character == 'space':
                typed_text += ' '
            elif predicted_character == 'backspace':
                typed_text = typed_text[:-1]
            elif predicted_character == 'enter':
                if typed_text.strip():
                    speak(typed_text)
                typed_text = ''
            else:
                typed_text += predicted_character
            stable_count = 0
            last_char = None

cap.release()
cv2.destroyAllWindows()
