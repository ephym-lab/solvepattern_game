import cv2
import math
import random
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. PUZZLE PIECE CLASS ---
class PuzzlePiece:
    def __init__(self, image_segment, correct_pos, start_pos, size):
        self.image = image_segment       # The cropped image patch (NumPy array)
        self.correct_pos = correct_pos   # (grid_col, grid_row) target position
        self.x = start_pos[0]           # Current pixel X coordinate on screen
        self.y = start_pos[1]           # Current pixel Y coordinate on screen
        self.size = size                 # (width, height) of the piece
        self.is_dragging = False         # Whether this piece is actively being held
        self.is_placed = False           # Whether this piece is correctly snapped

    def is_hovered(self, finger_x, finger_y):
        """Checks if the finger coordinates are inside this piece's bounding box."""
        return (self.x <= finger_x <= self.x + self.size[0] and
                self.y <= finger_y <= self.y + self.size[1])

# --- 2. PREPARE AND SLICE THE PUZZLE IMAGE ---
source_img = cv2.imread('puzzle_source.png')
if source_img is None:
    # Fallback: generate a colorful grid placeholder
    source_img = np.zeros((300, 300, 3), dtype=np.uint8)
    colors = [
        (200, 60, 60),   (60, 200, 60),   (60, 60, 200),
        (200, 200, 60),  (60, 200, 200),  (200, 60, 200),
        (150, 100, 50),  (50, 150, 100),  (100, 50, 150),
    ]
    tile = 100
    for i, color in enumerate(colors):
        r, c = divmod(i, 3)
        source_img[r*tile:(r+1)*tile, c*tile:(c+1)*tile] = color
        cv2.putText(source_img, str(i+1),
                    (c*tile + 35, r*tile + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

# Resize source image to a standard square size for predictability
PUZZLE_W, PUZZLE_H = 300, 300
source_img = cv2.resize(source_img, (PUZZLE_W, PUZZLE_H))

COLS, ROWS = 3, 3
PIECE_W = PUZZLE_W // COLS
PIECE_H = PUZZLE_H // ROWS

# --- 3. OPEN WEBCAM EARLY so we know frame dimensions for spawn math ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check that a camera is connected.")

# Grab one frame to get real resolution
ret, _probe = cap.read()
FRAME_H, FRAME_W = (_probe.shape[:2] if ret else (480, 640))

# Target drop zone anchored to top-right — computed ONCE outside the loop
TARGET_X = FRAME_W - PUZZLE_W - 20
TARGET_Y = 20
SNAP_TOL = 8   # pixel tolerance for win-check — fixes exact-equality bug

# Spawn zone is the LEFT side, safely away from the target drop zone
SPAWN_MAX_X = TARGET_X - PIECE_W - 10   # guaranteed no overlap with drop zone
SPAWN_MIN_X = 10
SPAWN_MIN_Y = PUZZLE_H + 30             # below the HUD area
SPAWN_MAX_Y = max(FRAME_H - PIECE_H - 10, SPAWN_MIN_Y + 10)

positions = []
for _ in range(ROWS * COLS):
    sx = random.randint(SPAWN_MIN_X, max(SPAWN_MIN_X, SPAWN_MAX_X))
    sy = random.randint(SPAWN_MIN_Y, max(SPAWN_MIN_Y, SPAWN_MAX_Y))
    positions.append((sx, sy))
random.shuffle(positions)

pieces = []
idx = 0
for r in range(ROWS):
    for c in range(COLS):
        crop_y1, crop_y2 = r * PIECE_H, (r + 1) * PIECE_H
        crop_x1, crop_x2 = c * PIECE_W, (c + 1) * PIECE_W
        segment = source_img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        pieces.append(PuzzlePiece(segment, (c, r), positions[idx], (PIECE_W, PIECE_H)))
        idx += 1

# --- 4. INITIALIZE MEDIAPIPE HAND LANDMARKER ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

# Colours
COL_SKELETON    = (255, 140,  0)
COL_TIP_IDLE    = (0,   0, 255)
COL_TIP_PINCH   = (0, 255,   0)
COL_PIECE_BORDER= (0, 255, 255)
COL_DRAG_BORDER = (0, 165, 255)   # orange highlight while held
COL_PLACED      = (0, 255,   0)   # green border when correctly placed
COL_GRID        = (180, 180, 180)
COL_TARGET_BOX  = (255, 255, 255)
COL_WIN_TEXT    = (0, 255,   0)

# Consecutive-frame camera-failure limit (fixes infinite-loop on dead cam)
MAX_CONSEC_FAILURES = 30
fail_count = 0

selected_pieces_by_hand: dict = {}

# --- 5. MAIN LOOP ---
try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            fail_count += 1
            if fail_count >= MAX_CONSEC_FAILURES:
                print("Camera unresponsive. Exiting.")
                break
            print(f"Empty frame ({fail_count}/{MAX_CONSEC_FAILURES}), retrying...")
            continue
        fail_count = 0

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # --- MediaPipe inference ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)

        # --- Draw target drop zone + inner grid lines ---
        cv2.rectangle(frame,
                      (TARGET_X, TARGET_Y),
                      (TARGET_X + PUZZLE_W, TARGET_Y + PUZZLE_H),
                      COL_TARGET_BOX, 2)
        for gc in range(1, COLS):
            lx = TARGET_X + gc * PIECE_W
            cv2.line(frame, (lx, TARGET_Y), (lx, TARGET_Y + PUZZLE_H), COL_GRID, 1)
        for gr in range(1, ROWS):
            ly = TARGET_Y + gr * PIECE_H
            cv2.line(frame, (TARGET_X, ly), (TARGET_X + PUZZLE_W, ly), COL_GRID, 1)

        # --- Parse landmarks ---
        active_hands: dict = {}
        if results.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
                for conn in HAND_CONNECTIONS:
                    pt1, pt2 = hand_landmarks[conn[0]], hand_landmarks[conn[1]]
                    x1, y1 = int(pt1.x * w), int(pt1.y * h)
                    x2, y2 = int(pt2.x * w), int(pt2.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), COL_SKELETON, 1)

                thumb = hand_landmarks[4]
                index = hand_landmarks[8]
                tx, ty = int(thumb.x * w), int(thumb.y * h)
                ix, iy = int(index.x * w), int(index.y * h)

                distance = math.hypot(ix - tx, iy - ty)
                pinching = distance < 35

                active_hands[hand_idx] = (ix, iy, pinching)

                dot_color = COL_TIP_PINCH if pinching else COL_TIP_IDLE
                cv2.circle(frame, (ix, iy), 8, dot_color, cv2.FILLED)
                cv2.circle(frame, (ix, iy), 8, (255, 255, 255), 1)  # white ring

        # --- Drag-and-drop engine ---
        for hand_idx, (ix, iy, pinching) in active_hands.items():
            if pinching:
                if hand_idx not in selected_pieces_by_hand:
                    for piece in pieces:
                        if (not piece.is_placed
                                and piece.is_hovered(ix, iy)
                                and piece not in selected_pieces_by_hand.values()):
                            selected_pieces_by_hand[hand_idx] = piece
                            piece.is_dragging = True
                            break
                else:
                    held = selected_pieces_by_hand[hand_idx]
                    held.x = ix - (held.size[0] // 2)
                    held.y = iy - (held.size[1] // 2)
            else:
                if hand_idx in selected_pieces_by_hand:
                    dropped = selected_pieces_by_hand.pop(hand_idx)
                    dropped.is_dragging = False

                    # Snap-check: is it close enough to the right cell?
                    grid_c = (dropped.x - TARGET_X + PIECE_W // 2) // PIECE_W
                    grid_r = (dropped.y - TARGET_Y + PIECE_H // 2) // PIECE_H

                    if 0 <= grid_c < COLS and 0 <= grid_r < ROWS:
                        if (grid_c, grid_r) == dropped.correct_pos:
                            dropped.x = TARGET_X + grid_c * PIECE_W
                            dropped.y = TARGET_Y + grid_r * PIECE_H
                            dropped.is_placed = True

        # Release pieces from hands that left the frame
        lost = [hid for hid in selected_pieces_by_hand if hid not in active_hands]
        for hid in lost:
            selected_pieces_by_hand[hid].is_dragging = False
            del selected_pieces_by_hand[hid]

        # --- Render puzzle pieces ---
        placed_count = 0
        for piece in pieces:
            py1 = max(0, piece.y)
            py2 = min(h, piece.y + piece.size[1])
            px1 = max(0, piece.x)
            px2 = min(w, piece.x + piece.size[0])
            ph, pw = py2 - py1, px2 - px1

            if ph > 0 and pw > 0:
                frame[py1:py2, px1:px2] = piece.image[0:ph, 0:pw]

                # Border colour encodes state
                if piece.is_placed:
                    border_col = COL_PLACED
                    placed_count += 1
                elif piece.is_dragging:
                    border_col = COL_DRAG_BORDER
                else:
                    border_col = COL_PIECE_BORDER
                cv2.rectangle(frame, (px1, py1), (px2, py2), border_col,
                              2 if piece.is_dragging else 1)

        # --- HUD: pieces placed counter ---
        total = ROWS * COLS
        hud_text = f"Placed: {placed_count}/{total}"
        cv2.rectangle(frame, (0, 0), (210, 40), (30, 30, 30), cv2.FILLED)
        cv2.putText(frame, hud_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        # --- Win condition: use tolerance, not exact equality ---
        # Also only trigger once all pieces are placed (not on first frame)
        all_correct = (placed_count == total)
        if all_correct:
            overlay = frame.copy()
            cv2.rectangle(overlay, (w//2 - 220, h//2 - 50),
                          (w//2 + 220, h//2 + 60), (20, 20, 20), cv2.FILLED)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, "PUZZLE SOLVED!", (w//2 - 185, h//2 + 15),
                        cv2.FONT_HERSHEY_DUPLEX, 1.6, COL_WIN_TEXT, 3)

        cv2.imshow("Multi-Hand Tracking Puzzle Engine", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # 'q' or ESC to quit
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    detector.close()