import time
import cv2
import numpy as np
from skimage.util import img_as_float, img_as_ubyte
from collections import deque

# ---------- Utilities ----------
def show_in_moved_window(win_name, img, x, y):
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name, img)

def _noop(v): pass

def _mk_controls_window(alpha_init=0.95, T_init=0.10, A_init=0.05, auto_init=0):
    cv2.namedWindow("Controls")
    cv2.resizeWindow("Controls", 420, 180)

    to_pos = lambda x: max(0, min(1000, int(round(x * 1000))))
    cv2.createTrackbar("Alpha (0..1)", "Controls", to_pos(alpha_init), 1000, _noop)
    cv2.createTrackbar("T (0..1)",     "Controls", to_pos(T_init),     1000, _noop)
    cv2.createTrackbar("A (0..1)",     "Controls", to_pos(A_init),     1000, _noop)
    cv2.createTrackbar("AutoReseedOnParamChange (0/1)", "Controls", auto_init, 1, _noop)

def _read_controls():
    pos_to_float = lambda p: float(p) / 1000.0
    alpha = pos_to_float(cv2.getTrackbarPos("Alpha (0..1)", "Controls"))
    T     = pos_to_float(cv2.getTrackbarPos("T (0..1)",     "Controls"))
    A     = pos_to_float(cv2.getTrackbarPos("A (0..1)",     "Controls"))
    auto  = cv2.getTrackbarPos("AutoReseedOnParamChange (0/1)", "Controls")

    alpha = max(0.0, min(alpha, 0.999))  # keep <1.0 so the EMA never freezes
    T     = max(0.0, min(T, 1.0))
    A     = max(0.0, min(A, 1.0))
    return alpha, T, A, int(auto)

# ---------- Main ----------
def capture_from_camera_and_show_images():
    print("Opening camera…")
    use_droid_cam = False
    cam = 0 if not use_droid_cam else "http://172.20.10.1:4747/video"
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    ok, frame_bgr = cap.read()
    if not ok:
        print("Can't receive frame")
        return

    H, W = frame_bgr.shape[:2]
    PIXELS = H * W
    FONT = cv2.FONT_HERSHEY_COMPLEX

    # Initial background model
    bg_gray = img_as_float(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))

    # Controls
    _mk_controls_window(alpha_init=0.95, T_init=0.10, A_init=0.05, auto_init=0)

    # For FPS and stats
    start_time = time.time()
    n_frames = 0
    avg_fg_pixels = deque(maxlen=60)

    # Auto-reseed-on-param-change bookkeeping
    prev_alpha, prev_T, prev_A, _ = _read_controls()
    param_changed = False
    QUIET_FG_THRESH = 0.02      # consider scene “quiet” if <2% FG pixels
    QUIET_FRAMES_NEEDED = 10    # quiet for this many consecutive frames
    quiet_counter = 0
    EPS = 1e-3                  # detect meaningful slider change

    print("Starting loop. Keys: q=quit, b=re-seed now.")
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("Can't receive frame. Exiting…")
            break

        alpha, T, A, auto_on = _read_controls()

        # Detect parameter change (any of alpha, T, A)
        if (abs(alpha - prev_alpha) > EPS or
            abs(T - prev_T) > EPS or
            abs(A - prev_A) > EPS):
            param_changed = True
            # snapshot current for next comparisons
            prev_alpha, prev_T, prev_A = alpha, T, A

        # Current frame -> grayscale float
        frame_gray = img_as_float(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))

        # Difference vs background
        dif_img = np.abs(frame_gray - bg_gray)

        # Threshold -> binary FG
        bin_img = dif_img > T

        # Stats
        F = int(np.sum(bin_img))
        Fp = F / PIXELS
        avg_fg_pixels.append(F)

        # FPS
        n_frames += 1
        elapsed = time.time() - start_time
        fps = int(n_frames / max(elapsed, 1e-6))

        # Optional auto-reseed after param changes, but only when scene is quiet
        if auto_on and param_changed:
            if Fp < QUIET_FG_THRESH:
                quiet_counter += 1
                if quiet_counter >= QUIET_FRAMES_NEEDED:
                    bg_gray = frame_gray.copy()
                    param_changed = False
                    quiet_counter = 0
                    print("Auto re-seeded background after parameter change (scene quiet).")
            else:
                quiet_counter = 0

        # Alarm overlay
        if Fp > A:
            cv2.putText(
                frame_bgr, "Change Detected!",
                (W // 2 - 150, H // 2), FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA
            )

        # HUD overlays
        cv2.putText(frame_bgr, f"fps: {fps}", (20, 40), FONT, 1, (255, 255, 255), 1)
        cv2.putText(
            frame_bgr,
            f"FG: {Fp*100:.1f}%  (T={T:.3f}, A={A:.3f}, a={alpha:.3f})",
            (20, 80), FONT, 0.7, (255, 255, 255), 1
        )
        cv2.putText(
            frame_bgr,
            f"AutoReseed:{'ON' if auto_on else 'OFF'}  Quiet<{QUIET_FG_THRESH*100:.0f}% for {QUIET_FRAMES_NEEDED} frames",
            (20, 115), FONT, 0.6, (0, 255, 0), 1 # COLOR=BGR
        )

        # Binary to uint8 for display
        bin_img_uint8 = img_as_ubyte(bin_img)

        # Show views
        show_in_moved_window("Input", frame_bgr, 0, 10)
        show_in_moved_window("Background image", bg_gray, W, 10)
        show_in_moved_window("Difference image", dif_img, 0, H + 10)
        show_in_moved_window("Binary image", bin_img_uint8, W, H + 10)

        # Background EMA update
        bg_gray = alpha * bg_gray + (1.0 - alpha) * frame_gray

        # Keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            bg_gray = frame_gray.copy()
            param_changed = False
            quiet_counter = 0
            print("Background re-seeded (manual).")

        # console line
        print(f"F={F:7d}  FG%={Fp*100:6.2f}  fps={fps:3d}  auto={auto_on}      ", end="\r")

    print("\nStopping.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_from_camera_and_show_images()
