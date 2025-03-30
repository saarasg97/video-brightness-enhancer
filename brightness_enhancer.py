import cv2
import numpy as np

def enhance_brightness_and_color(frame):
    """Enhance both brightness and color (saturation)."""

    # --- Step 1: Brightness Enhancement using CLAHE ---
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    lab_enhanced = cv2.merge((cl, a, b))
    bright_frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # --- Step 2: Color Enhancement in HSV (Saturation Boost) ---
    hsv = cv2.cvtColor(bright_frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Boost saturation (increase by 30, max 255)
    s = cv2.add(s, 30)
    s = np.clip(s, 0, 255)

    hsv_enhanced = cv2.merge((h, s, v))
    color_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    # --- Optional Step 3: Slight Sharpening ---
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(color_enhanced, -1, kernel)

    return sharpened

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        enhanced = enhance_brightness_and_color(frame)

        cv2.imshow("Original", frame)
        cv2.imshow("Enhanced Brightness + Color", enhanced)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
