import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import threading

# Initialize MediaPipe Hands with optimized settings
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Add a lock to prevent key actions when window is not focused
game_window_active = False
action_lock = threading.Lock()

# Track key states
key_pressed = {
    'w': False, 's': False, 'a': False, 'd': False,
    'shift': False, 'ctrl': False, 'e': False, 'rmb': False
}

# For gesture timing and debounce
last_action_time = time.time()
DEBOUNCE_TIME = 0.3

# Frame processing
PROCESS_EVERY_N_FRAMES = 1
frame_count = 0

# Display configuration - increased size
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Input resolution - higher quality camera feed
INPUT_WIDTH = 800
INPUT_HEIGHT = 450

# Hand position tracking
hand_history = []
MAX_HISTORY = 5

# Tilt thresholds - adjust these as needed
TILT_THRESHOLD = 0.15
WRIST_TILT_BUFFER = []
WRIST_TILT_BUFFER_SIZE = 3

# Track current gesture to prevent overlaps
current_gesture = None
gesture_history = []
GESTURE_STABILITY = 3  # Number of frames for stable gesture

# Webcam feed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_HEIGHT)

# Function to release all keys
def release_all_keys():
    with action_lock:
        for key in key_pressed:
            if key_pressed[key] and key != 'rmb':
                pyautogui.keyUp(key)
                key_pressed[key] = False
        if key_pressed['rmb']:
            pyautogui.mouseUp(button='right')
            key_pressed['rmb'] = False

# Function to safely press keys only when game window is active
def safe_key_press(key, press=True):
    global key_pressed
    if not game_window_active:
        # Only show visual feedback but don't actually press keys
        return
        
    with action_lock:
        if press:
            if not key_pressed[key]:
                if key == 'rmb':
                    pyautogui.mouseDown(button='right')
                else:
                    pyautogui.keyDown(key)
                key_pressed[key] = True
        else:
            if key_pressed[key]:
                if key == 'rmb':
                    pyautogui.mouseUp(button='right')
                else:
                    pyautogui.keyUp(key)
                key_pressed[key] = False

# Function for one-time key presses
def safe_key_tap(key):
    if not game_window_active:
        # Only show visual feedback but don't actually press keys
        return
        
    with action_lock:
        pyautogui.press(key)

# MOVEMENT CONTROL: Hand position in frame
def detect_hand_position(landmarks, frame_height, frame_width):
    """Detect hand position in the frame for movement"""
    palm_center = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    
    # Convert normalized coordinates to frame coordinates
    x_pos = palm_center.x
    y_pos = palm_center.y
    
    # Define frame regions for movement - smaller center zone for better control
    center_x_min, center_x_max = 0.4, 0.6  # Smaller center zone
    center_y_min, center_y_max = 0.4, 0.6  # Smaller center zone
    
    # Determine movement direction based on hand position
    x_dir = None
    if x_pos < center_x_min:
        x_dir = 'left'
    elif x_pos > center_x_max:
        x_dir = 'right'
        
    y_dir = None
    if y_pos < center_y_min:
        y_dir = 'up'
    elif y_pos > center_y_max:
        y_dir = 'down'
    
    return x_dir, y_dir

# Check if hand is closed (fist)
def is_hand_closed(landmarks):
    """Detect closed fist - all fingers must be curled"""
    finger_tips = [
        landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
        landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    ]
    
    finger_mcps = [
        landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
        landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
        landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
        landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    ]
    
    # For a true fist, all fingers must be curled
    for tip, mcp in zip(finger_tips, finger_mcps):
        if tip.y < mcp.y:  # If any tip is higher than MCP, it's not a fist
            return False
    
    return True

# Check for thumb and pinky pinch (for crouch)
def is_thumb_pinky_pinch(landmarks):
    """Detect thumb and pinky pinch - thumb must be near pinky tip"""
    # Get finger tips and bases
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    
    # MCPs for reference
    index_mcp = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    
    # Distance between thumb and pinky
    distance = np.sqrt((thumb_tip.x - pinky_tip.x)**2 + (thumb_tip.y - pinky_tip.y)**2)
    
    # Other fingers should be extended for clearer distinction
    other_fingers_extended = (
        index_tip.y < index_mcp.y and
        middle_tip.y < middle_mcp.y and
        ring_tip.y < ring_mcp.y
    )
    
    return distance < 0.07 and other_fingers_extended  # Pinch + other fingers extended

# Check for thumb and index finger pinch (interact)
def is_thumb_index_pinch(landmarks):
    """Detect thumb and index pinch - thumb must be near index tip"""
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # MCPs for reference
    middle_mcp = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    
    # Distance between thumb and index
    distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    
    # Ensure other fingers are curled to distinguish from other gestures
    other_fingers_curled = (
        middle_tip.y > middle_mcp.y and
        ring_tip.y > ring_mcp.y and
        pinky_tip.y > pinky_mcp.y
    )
    
    return distance < 0.05 and other_fingers_curled  # Pinch + other fingers curled

# Detect pointing gesture for aiming
def is_pointing(landmarks):
    """Detect pointing - index extended, all other fingers curled"""
    # Get finger tips
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    
    # Get finger bases
    index_mcp = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    thumb_mcp = landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    
    # Check if index is extended and other fingers are curled
    index_extended = index_tip.y < index_mcp.y - 0.05  # Ensure it's clearly extended
    
    others_curled = (
        middle_tip.y > middle_mcp.y and
        ring_tip.y > ring_mcp.y and
        pinky_tip.y > pinky_mcp.y and
        thumb_tip.x > thumb_mcp.x  # Thumb tucked in
    )
    
    return index_extended and others_curled

# NEW: Detect stable gesture
def get_stable_gesture(gesture, landmarks, frame_height, frame_width):
    global gesture_history
    
    # Add current gesture to history
    gesture_history.append(gesture)
    if len(gesture_history) > GESTURE_STABILITY:
        gesture_history.pop(0)
    
    # Only return a gesture if it's stable for GESTURE_STABILITY frames
    if len(gesture_history) == GESTURE_STABILITY and all(g == gesture_history[0] for g in gesture_history):
        return gesture_history[0]
    
    return None

try:
    # Create window with specific size
    cv2.namedWindow("Plague Tale Hand Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Plague Tale Hand Controls", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            continue

        # Flip horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        
        # Resize frame for display
        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        visual_frame = display_frame.copy()
        
        # Add title header - larger text
        cv2.putText(visual_frame, "Plague Tale Hand Controls", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)  # Bigger shadow
        cv2.putText(visual_frame, "Plague Tale Hand Controls", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)  # Bigger text
        
        # Add activation status - larger text
        status_color = (0, 255, 0) if game_window_active else (0, 0, 255)
        status_text = "CONTROLS ACTIVE" if game_window_active else "CONTROLS INACTIVE (Press SPACE to toggle)"
        cv2.putText(visual_frame, status_text, (DISPLAY_WIDTH - 500, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)  # Bigger shadow
        cv2.putText(visual_frame, status_text, (DISPLAY_WIDTH - 500, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)  # Bigger text
        
        # Process hand detection
        frame_count += 1
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Reset keys if no hand is detected
            if not results.multi_hand_landmarks:
                release_all_keys()
                hand_history.clear()
                gesture_history.clear()
                current_gesture = None
            
            # Process hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get key landmarks for visualization
                    palm_center = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    
                    # Store hand position history
                    hand_history.append((palm_center.x, palm_center.y, palm_center.z))
                    if len(hand_history) > MAX_HISTORY:
                        hand_history.pop(0)
                    
                    # Draw hand landmarks - thicker lines for better visibility
                    mp_draw.draw_landmarks(
                        visual_frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),  # Thicker
                        mp_draw.DrawingSpec(color=(255, 0, 0), thickness=3)  # Thicker
                    )
                    
                    # Detect gestures - each gesture is mutually exclusive
                    is_closed_hand = is_hand_closed(hand_landmarks)
                    is_thumb_index = is_thumb_index_pinch(hand_landmarks)
                    is_thumb_pinky = is_thumb_pinky_pinch(hand_landmarks)
                    is_point_gesture = is_pointing(hand_landmarks)
                    
                    # NEW: Get hand position in frame for movement
                    x_dir, y_dir = detect_hand_position(hand_landmarks, DISPLAY_HEIGHT, DISPLAY_WIDTH)
                    
                    # Determine current gesture - prioritize specific gestures
                    if is_closed_hand:
                        current_gesture = "FIST"
                    elif is_thumb_index:
                        current_gesture = "THUMB_INDEX"
                    elif is_thumb_pinky:
                        current_gesture = "THUMB_PINKY"
                    elif is_point_gesture:
                        current_gesture = "POINTING"
                    else:
                        current_gesture = "NEUTRAL"
                    
                    # Get stable gesture to prevent flickering
                    stable_gesture = get_stable_gesture(current_gesture, hand_landmarks, DISPLAY_HEIGHT, DISPLAY_WIDTH)
                    
                    # Status display
                    status_text = []
                    
                    # Draw position guides
                    # Center zone
                    cv2.rectangle(visual_frame, 
                                 (int(0.4*DISPLAY_WIDTH), int(0.4*DISPLAY_HEIGHT)), 
                                 (int(0.6*DISPLAY_WIDTH), int(0.6*DISPLAY_HEIGHT)), 
                                 (255, 255, 255), 1)
                    
                    # ----- MOVEMENT CONTROLS (WASD) based on hand position in frame -----
                    
                    # Only apply movement if we're in NEUTRAL gesture or FIST (for sprint)
                    can_move = stable_gesture in ["NEUTRAL", "FIST", None]
                    
                    # Forward/Backward (W/S) based on hand position in frame
                    if can_move and y_dir == 'up':
                        safe_key_press('w', True)
                        status_text.append("W - Forward")
                    else:
                        safe_key_press('w', False)
                    
                    if can_move and y_dir == 'down':
                        safe_key_press('s', True)
                        status_text.append("S - Backward")
                    else:
                        safe_key_press('s', False)
                    
                    # Left/Right (A/D) based on hand position in frame
                    if can_move and x_dir == 'left':
                        safe_key_press('a', True)
                        status_text.append("A - Left")
                    else:
                        safe_key_press('a', False)
                    
                    if can_move and x_dir == 'right':
                        safe_key_press('d', True)
                        status_text.append("D - Right")
                    else:
                        safe_key_press('d', False)
                    
                    # ----- SPECIAL ACTIONS -----
                    
                    # Sprint (Shift) - closed fist while moving
                    if stable_gesture == "FIST" and (x_dir or y_dir):
                        safe_key_press('shift', True)
                        status_text.append("SHIFT - Sprint")
                    else:
                        safe_key_press('shift', False)
                    
                    # Crouch (Ctrl) - thumb to pinky pinch
                    if stable_gesture == "THUMB_PINKY":
                        safe_key_press('ctrl', True)
                        status_text.append("CTRL - Crouch")
                    else:
                        safe_key_press('ctrl', False)
                    
                    # Interact (E) - thumb to index finger pinch
                    current_time = time.time()
                    if stable_gesture == "THUMB_INDEX" and current_time - last_action_time > DEBOUNCE_TIME:
                        safe_key_tap('e')
                        last_action_time = current_time
                        status_text.append("E - Interact")
                    
                    # Aim (RMB) - pointing gesture (index finger extended)
                    if stable_gesture == "POINTING":
                        safe_key_press('rmb', True)
                        status_text.append("RMB - Aim")
                    else:
                        safe_key_press('rmb', False)
                    
                    # Add gesture information text
                    gesture_text = []
                    if stable_gesture:
                        gesture_text.append(f"GESTURE: {stable_gesture}")
                    if x_dir:
                        gesture_text.append(f"X: {x_dir.upper()}")
                    if y_dir:
                        gesture_text.append(f"Y: {y_dir.upper()}")
                    
                    # Display hand position & gesture info - larger text
                    cv2.putText(visual_frame, " | ".join(gesture_text), (20, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)  # Bigger shadow
                    cv2.putText(visual_frame, " | ".join(gesture_text), (20, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)  # Bigger text
                    
                    # Display status text - larger text
                    y_pos = 150
                    for status in status_text:
                        cv2.putText(visual_frame, status, (20, y_pos), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
                        cv2.putText(visual_frame, status, (20, y_pos), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        y_pos += 40
        
        # Add control guide overlay - larger overlay
        overlay = visual_frame.copy()
        cv2.rectangle(overlay, (10, DISPLAY_HEIGHT - 240), (550, DISPLAY_HEIGHT - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, visual_frame, 0.3, 0, visual_frame)
        cv2.rectangle(visual_frame, (10, DISPLAY_HEIGHT - 240), (550, DISPLAY_HEIGHT - 10), (255, 255, 255), 1)
        
        # Draw grid zones for movement - adjusted for larger display
        cv2.putText(visual_frame, "Movement Zones:", (600, DISPLAY_HEIGHT - 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        zone_display = visual_frame[DISPLAY_HEIGHT-195:DISPLAY_HEIGHT-25, 600:850]
        cv2.rectangle(zone_display, (35, 35), (95, 95), (255, 255, 255), 1)  # Center zone
        # Draw arrows for directions - bigger text
        cv2.putText(zone_display, "W", (65, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(zone_display, "S", (65, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(zone_display, "A", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(zone_display, "D", (125, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Control instructions
        controls = [
            "CONTROLS:",
            "Press SPACE to toggle controls ON/OFF",
            "Move Hand Outside Center Box → WASD movement",
            "Make a Fist While Moving → Sprint (Shift)",
            "Thumb + Pinky Pinch (others extended) → Crouch (Ctrl)",
            "Thumb + Index Pinch (others curled) → Interact (E)",
            "Point with Index (others curled) → Aim (RMB)",
            "Press Q to quit"
        ]
        
        # Control instructions - larger text
        y_offset = DISPLAY_HEIGHT - 220
        for line in controls:
            cv2.putText(visual_frame, line, (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # Thicker text
            y_offset += 28  # More space between lines
        
        # Display frame with configured window size
        cv2.imshow("Plague Tale Hand Controls", visual_frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Toggle control activation with spacebar
            game_window_active = not game_window_active
            if not game_window_active:
                release_all_keys()  # Release all keys when deactivating

except Exception as e:
    print(f"Error: {e}")
    
finally:
    release_all_keys()
    cap.release()
    cv2.destroyAllWindows()
