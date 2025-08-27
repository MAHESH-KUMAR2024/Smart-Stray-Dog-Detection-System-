# main.py
import threading
import csv
import time
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, filedialog
from ultralytics import YOLO

# -----------------------
# Configuration
# -----------------------
device = "mps" if hasattr(__import__('torch').backends, 'mps') and __import__('torch').backends.mps.is_available() \
         else "cuda" if __import__('torch').cuda.is_available() else "cpu"
print("Device:", device)

# Put your video filenames here (or allow user to change)
video_files = ["dogs.mp4", "dog_walk_vid.mp4"]
DOG_CLASS_ID = 16   # COCO dog class id

# -----------------------
# Model (YOLOv8)
# -----------------------
model = YOLO("yolov8m.pt")  # change path if needed

# -----------------------
# Shared state (thread-safe access required)
# -----------------------
latest_frame = None
latest_frame_lock = threading.Lock()

running = False
stop_flag = False
next_flag = False
current_video_index = 0

# Store unique IDs per file: { filename: set(unique_keys) }
per_video_seen = {}

# When detection completes, this will be used to save CSV
report_filename = "dog_report.csv"

# -----------------------
# Helper: create unique key for detection
# - If tracker id exists: ("id", track_id)
# - Else fallback signature: ("sig", cx//grid, cy//grid)
# -----------------------
def make_unique_key(track_id, bbox, grid=50):
    if track_id is not None:
        return ("id", int(track_id))
    else:
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) // 2)
        cy = int((y1 + y2) // 2)
        return ("sig", cx // grid, cy // grid)

# -----------------------
# Detection thread
# Uses model.track to get stable track IDs (ByteTrack).
# Puts the latest visual frame into latest_frame for the GUI main thread to render.
# -----------------------
def detection_thread():
    global running, stop_flag, next_flag, current_video_index, latest_frame, per_video_seen

    running = True
    stop_flag = False
    next_flag = False
    per_video_seen = {}

    while current_video_index < len(video_files) and not stop_flag:
        video_path = video_files[current_video_index]
        per_video_seen.setdefault(video_path, set())
        # Use tracker — this yields a stream of results with track IDs in boxes.id (when available)
        # tracker="bytetrack.yaml" is commonly used; ultralytics provides builtin configs
        try:
            tracker_stream = model.track(
                source=video_path,
                stream=True,
                device=device,
                tracker="bytetrack.yaml",
                persist=True,
                classes=[DOG_CLASS_ID],
                conf=0.25, iou=0.45
            )
        except Exception as e:
            # If model.track fails (older ultralytics versions), fallback to frame-by-frame detection without tracking
            print("model.track failed, falling back to frame-by-frame detection:", e)
            tracker_stream = None

        if tracker_stream is not None:
            # Streamed tracking loop
            for result in tracker_stream:
                if stop_flag or next_flag:
                    break

                # result.orig_img is the BGR frame
                frame = result.orig_img.copy()
                boxes = getattr(result, "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy().astype(int)
                    # boxes.id may or may not be present; handle safely
                    ids_arr = None
                    try:
                        ids_tensor = boxes.id
                        if ids_tensor is not None:
                            ids_arr = ids_tensor.cpu().numpy()
                    except Exception:
                        ids_arr = None

                    for i, (x1, y1, x2, y2) in enumerate(xyxy):
                        track_id = None
                        if ids_arr is not None:
                            # some tracker implementations may set id = -1 for unknown; allow it but treat negative as None
                            val = int(ids_arr[i])
                            if val >= 0:
                                track_id = val
                        unique_key = make_unique_key(track_id, (x1, y1, x2, y2))
                        if unique_key not in per_video_seen[video_path]:
                            per_video_seen[video_path].add(unique_key)

                        # Draw box and label
                        label = "Dog"
                        if unique_key[0] == "id":
                            label = f"Dog #{unique_key[1]}"
                        else:
                            label = "Dog (s)"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Write counts overlay
                cur_count = len(per_video_seen[video_path])
                total = sum(len(v) for v in per_video_seen.values())
                cv2.putText(frame, f"In-file: {cur_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(frame, f"Overall: {total}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(frame, f"Press 'Next' or 'n' to skip", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)

                # Publish latest frame (thread-safe)
                with latest_frame_lock:
                    latest_frame = frame.copy()

                # Sleep tiny bit to let GUI render; also avoid hammering CPU
                time.sleep(0.01)

            # end for result
            # reset next_flag if it triggered a skip
            if next_flag:
                next_flag = False
                current_video_index += 1
                continue

        else:
            # Fallback: frame-by-frame detection (no tracker ids)
            cap = cv2.VideoCapture(video_path)
            grid = 50
            while cap.isOpened() and not stop_flag and not next_flag:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame, device=device, classes=[DOG_CLASS_ID], conf=0.25)
                result = results[0]
                boxes = result.boxes
                if boxes is not None and len(boxes)>0:
                    xyxy = boxes.xyxy.cpu().numpy().astype(int)
                    for (x1,y1,x2,y2) in xyxy:
                        unique_key = make_unique_key(None, (x1,y1,x2,y2), grid=grid)
                        if unique_key not in per_video_seen[video_path]:
                            per_video_seen[video_path].add(unique_key)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                        cv2.putText(frame, "Dog (s)", (x1, max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cur_count = len(per_video_seen[video_path])
                total = sum(len(v) for v in per_video_seen.values())
                cv2.putText(frame, f"In-file: {cur_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(frame, f"Overall: {total}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                with latest_frame_lock:
                    latest_frame = frame.copy()
                time.sleep(0.03)

            cap.release()
            if next_flag:
                next_flag = False
                current_video_index += 1
                continue

        # when stream ended normally, move to next file
        current_video_index += 1

    # detection loop ended
    running = False
    # create the CSV report
    try:
        save_report_csv(report_filename, per_video_seen)
    except Exception as e:
        print("Failed saving CSV:", e)
    # show final GUI popup from main thread using after
    root.after(0, show_final_report)

# -----------------------
# Save report CSV
# -----------------------
def save_report_csv(csv_name, per_video_seen_map):
    total = sum(len(v) for v in per_video_seen_map.values())
    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "unique_dogs"])
        for fname, s in per_video_seen_map.items():
            writer.writerow([fname, len(s)])
        writer.writerow([])
        writer.writerow(["TOTAL", total])
    print("Saved report to", csv_name)

# -----------------------
# GUI refresh loop
# Runs in main thread via root.after
# -----------------------
def refresh_gui():
    # draw latest_frame if exists
    with latest_frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None

    if frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_panel.imgtk = imgtk
        video_panel.configure(image=imgtk)

    # update report text
    report_text.delete("1.0", tk.END)
    total = sum(len(s) for s in per_video_seen.values())
    for fname in video_files:
        count = len(per_video_seen.get(fname, set()))
        report_text.insert(tk.END, f"{fname}: {count}  \n")
    report_text.insert(tk.END, f"\nOverall total: {total}\n")
    report_text.insert(tk.END, "\nControls: Start | Stop | Next (or press 'n')\n")

    root.after(100, refresh_gui)

# -----------------------
# GUI actions
# -----------------------
def start_button_cb():
    global running, stop_flag, current_video_index, per_video_seen, latest_frame
    if running:
        return
    # reset state
    stop_flag = False
    current_video_index = 0
    per_video_seen = {}
    with latest_frame_lock:
        latest_frame = None
    # start detection thread
    t = threading.Thread(target=detection_thread, daemon=True)
    t.start()

def stop_button_cb():
    global stop_flag
    stop_flag = True

def next_button_cb():
    global next_flag
    if not running:
        # if not running just advance index visually
        advance_nonrunning_video()
        return
    next_flag = True

def advance_nonrunning_video():
    global current_video_index
    if current_video_index + 1 < len(video_files):
        current_video_index += 1
    refresh_report_once()

def refresh_report_once():
    # update report_text immediately
    report_text.delete("1.0", tk.END)
    total = sum(len(s) for s in per_video_seen.values())
    for fname in video_files:
        count = len(per_video_seen.get(fname, set()))
        report_text.insert(tk.END, f"{fname}: {count}  \n")
    report_text.insert(tk.END, f"\nOverall total: {total}\n")

def on_key_press(event):
    if event.char == 'n':
        next_button_cb()

def show_final_report():
    # show final report window and popup, called in main thread
    total = sum(len(s) for s in per_video_seen.values())
    txt = ""
    for fname in video_files:
        txt += f"{fname}: {len(per_video_seen.get(fname, set()))} dogs\n"
    txt += f"\nTOTAL: {total} dogs\n\nReport saved to: {report_filename}"
    # popup
    messagebox.showinfo("Final Report", txt)
    # Also create a Toplevel with the text
    w = tk.Toplevel(root)
    w.title("Final Dog Detection Report")
    t = tk.Text(w, width=60, height=20, font=("Courier", 11))
    t.pack(fill="both", expand=True)
    t.insert("1.0", txt)

# -----------------------
# Build GUI
# -----------------------
root = tk.Tk()
root.title("Dog Detection & Tracking (unique-count)")
root.geometry("1100x650")

# Left control frame
ctrl = tk.Frame(root, width=220, bg="#ddd")
ctrl.pack(side="left", fill="y")

start_btn = tk.Button(ctrl, text="Start", width=18, height=2, bg="green", fg="white", command=start_button_cb)
start_btn.pack(pady=16)

stop_btn = tk.Button(ctrl, text="Stop", width=18, height=2, bg="red", fg="white", command=stop_button_cb)
stop_btn.pack(pady=6)

next_btn = tk.Button(ctrl, text="Next Video", width=18, height=2, command=next_button_cb)
next_btn.pack(pady=6)

# Option to select videos (optional)
def add_videos_cb():
    files = filedialog.askopenfilenames(title="Select video files", filetypes=[("MP4 files","*.mp4"),("All files","*.*")])
    if files:
        global video_files
        video_files = list(files)
        refresh_report_once()

add_vid_btn = tk.Button(ctrl, text="Add Videos...", width=18, command=add_videos_cb)
add_vid_btn.pack(pady=6)

# Center: video preview
video_panel = tk.Label(root, bg="black")
video_panel.pack(side="left", expand=True, fill="both", padx=6, pady=6)

# Right: report area
report_frame = tk.Frame(root, width=260)
report_frame.pack(side="right", fill="y")

lbl = tk.Label(report_frame, text="Live Report", font=("Helvetica", 14))
lbl.pack(pady=6)
report_text = tk.Text(report_frame, width=36, height=30, font=("Arial", 11))
report_text.pack(padx=6, pady=6)

# Bind key
root.bind("<Key>", on_key_press)

# Start GUI refresher
root.after(100, refresh_gui)

root.mainloop()
