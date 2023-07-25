import cv2
import os


def load_frame_sequences(folder_path, sequence_duration, image_extensions=["jpg"], frame_rate=25):
    # Get sorted list of image files
    image_files = sorted([
        file for file in os.listdir(folder_path)
        if os.path.splitext(file)[1].lower() in image_extensions
    ])

    # Create a dictionary to store frame IDs and corresponding frames
    frames = {}
    for image_file in image_files:
        frame_id = int(os.path.splitext(image_file)[0])
        frame_path = os.path.join(folder_path, image_file)
        frames[frame_id] = frame_path

    # Group frames into sequences
    sequences = []
    current_sequence = []
    previous_frame = None
    for frame_id in sorted(frames.keys()):
        frame_path = frames[frame_id]

        if previous_frame is not None:
            frame_time_diff = (frame_id - previous_frame) / frame_rate
            if frame_time_diff > sequence_duration:
                sequences.append(current_sequence)
                current_sequence = []
            elif frame_time_diff > 1:
                # Fill missing frames with the previous frame
                missing_frames = int((frame_time_diff - 1) * frame_rate)
                for _ in range(missing_frames):
                    current_sequence.append(previous_frame)

        current_sequence.append(frame_path)
        previous_frame = frame_id

    # Add the last sequence
    sequences.append(current_sequence)

    # Process each sequence as needed
    for sequence in sequences:
        # Perform further processing on the frames within the sequence
        for frame_path in sequence:
            frame = cv2.imread(frame_path)
            # Process the frame as desired
