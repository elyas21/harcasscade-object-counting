import os
import cv2
import numpy as np
import datetime
import torch
from absl import app, flags
from deep_sort_realtime.deepsort_tracker import DeepSort


def main(_argv):
    # Open a video capture object
    video_path = 'm1080.mp4'
    cap = cv2.VideoCapture(video_path)

    frame_skip = 10  # Skip every 10 frames
    frame_count = 0

    face_list = []
    # Initialize the DeepSORT tracker
    tracker = DeepSort(max_age=50)

    # Define rectangular region coordinates (x1, y1) top-left and (x2, y2) bottom-right
    rect_x1, rect_y1 = 500, 100  # Example coordinates, adjust as per your video frame size
    rect_x2, rect_y2 = 1000, 400  # Example coordinates, adjust as per your video frame size

    while True:
        # Start time to compute the FPS
        start = datetime.datetime.now()

        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            print("End of the video file...")
            break
        
        frame_count += 1

        # Skip frames if not a multiple of frame_skip
        if frame_count % frame_skip != 0:
            continue

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform face detection using Haar Cascade
        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Prepare detections for DeepSORT
        detections = []
        for (x, y, w, h) in faces:
            # Calculate center of the detected face
            center_x = x + w // 2
            center_y = y + h // 2

            # Check if the center of the face is within the rectangular region
            if rect_x1 <= center_x <= rect_x2 and rect_y1 <= center_y <= rect_y2:
                bbox = (x, y, x + w, y + h)
                confidence = 1  # Confidence score for the detection
                detections.append([bbox, confidence, 0])  # Third element is class_id (0 for face)

        # Update DeepSORT tracker with current detections
        tracks = tracker.update_tracks(detections, frame=frame)

        # Loop over the tracks
        for track in tracks:
            # If the track is not confirmed, ignore it
            if not track.is_confirmed():
                continue

            # Get the track ID and the bounding box
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

            # Draw bounding box and track ID on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Face {track_id}', (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            print(track)
            # if track.num_frames == 1:
            if track.track_id  not in face_list: 
                face_list.append(track.track_id)
                # Save the first frame as an image
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join('output', f'{timestamp}_face{track_id}.jpg')
                cv2.imwrite(filename, frame)

        # End time to compute the FPS
        end = datetime.datetime.now()

        # Calculate the frames per second and draw it on the frame
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

        # Draw the rectangular region on the frame (optional for visualization)
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 0, 0), 2)



        cv2.imshow("Face Tracking with DeepSORT", frame)

        # Check for 'q' key press to exit the loop
        if cv2.waitKey(1) == ord("q"):
            break

    # Release video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Parse command line flags and execute the main function
    app.run(main)
