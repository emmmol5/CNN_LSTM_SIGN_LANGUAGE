from IMPORTS import *
metadata = r"C:\Users\Emma\OneDrive - Noroff Education AS\3. Året\Bachelor\Sign Language Health\helse_ordliste_mod.xlsx"
aug_folder = r"C:\Users\Emma\OneDrive - Noroff Education AS\3. Året\Bachelor\Sign Language Health\aug_videos"
zero_pad_resized = r"C:\Users\Emma\OneDrive - Noroff Education AS\3. Året\Bachelor\Sign Language Health\vid_zeropad_resized"

df = pd.read_excel(metadata)
os.makedirs(aug_folder, exist_ok=True)

def change_speed(video, factor):
    return video.fx(mp.vfx.speedx, factor)

def adjust_brightness(video, factor):
    return video.fl_image(lambda frame: np.clip(frame * factor, 0, 255).astype(np.uint8))

def rotate_video(video, angle):
    return video.rotate(angle)

def crop_and_resize(video, crop_factor=0.9):
    w, h = video.size
    return video.crop(x1=w*(1-crop_factor)//2, y1=h*(1-crop_factor)//2, x2=w*(1+crop_factor)//2, y2=h*(1+crop_factor)//2).resize((w, h))

def horizontal_flip(video):
    return video.fx(mp.vfx.mirror_x)

def adjust_contrast(video, factor):
    return video.fx(mp.vfx.lum_contrast, contrast=factor)

def process_augmentation():
    for _, row in df.iterrows():
        term = row["Health_Term"]
        video_file = row["Video_File"]
        
        input_path = os.path.join(video_folder, video_file)
        term_folder = os.path.join(aug_folder, term.replace(".mp4", ""))
        os.makedirs(term_folder, exist_ok=True)
        
        # Load video
        video = mp.VideoFileClip(input_path)
        
        # Save original
        video.write_videofile(os.path.join(term_folder, video_file), codec="libx264", audio=False)
        
        augmentations = [
            (change_speed, (0.8,)),
            (change_speed, (1.2,)),
            (adjust_brightness, (1.2,)),
            (adjust_brightness, (0.8,)),
            (rotate_video, (5,)),
            (rotate_video, (-5,)),
            (crop_and_resize, (0.85,)),
            (horizontal_flip, ()),
            (adjust_contrast, (1.2,)),   
        ]
        
        for i, (func, args) in enumerate(augmentations):
            aug_video = func(video, *args)
            output_path = os.path.join(term_folder, f"aug_{i+1}_" + video_file)
            aug_video.write_videofile(output_path, codec="libx264", audio=False)
        
        print(f"Processed {term}: {video_file}")


# Frame Processing Functions
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        frames.append(frame)
    cap.release()
    return np.array(frames)  


target_size = (128, 128)
target_frames = 117  

def resize_frames(video, target_size):
    resized_frames = [cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4) for frame in video]
    print(f"Resized {len(resized_frames)} frames")
    return resized_frames

def zero_pad_frames(frames, target_frames):
    frame_count = len(frames)
    print(f"Original frame count: {frame_count}")

    if frame_count < target_frames:
        padding = [np.zeros_like(frames[0], dtype=np.uint8) for _ in range(target_frames - frame_count)]
        result = frames + padding
        print(f"Padded to: {len(result)} frames") 
        return result
    elif frame_count > target_frames:
        print(f"WARNING: Video has more than {target_frames} frames. Trimming to {target_frames} frames.")
        return frames[:target_frames] 
    else:
        print(f"No padding needed, video already has {frame_count} frames.")
        return frames


def save_video_to_folder(video_frames, output_path, term_name, video_id, fps=25):
    term_folder = os.path.join(output_path, term_name)
    os.makedirs(term_folder, exist_ok=True)
    save_path = os.path.join(term_folder, f"{video_id}.mp4")

    height, width, _ = video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))  
    
    for frame in video_frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Saved: {save_path}")


def process_resize_pad(aug_folder, output_folder, target_size, target_frames):
    total_videos = 0
    for term_name in os.listdir(aug_folder):
        term_path = os.path.join(aug_folder, term_name)
        if os.path.isdir(term_path):
            for video_file in os.listdir(term_path):
                if video_file.endswith('.mp4'):
                    video_path = os.path.join(term_path, video_file)
                    frames = load_video(video_path)

                    if len(frames) == 0:
                        print(f"ERROR: Could not read frames from {video_path}")
                        continue

                    print(f"Processing video: {video_file}, Frame count before resize: {len(frames)}")

                    resized = resize_frames(frames, target_size)
                    padded = zero_pad_frames(resized, target_frames)

                    if len(padded) != target_frames:
                        print(f"WARNING: {video_path} ended up with {len(padded)} frames! Skipping this video.")
                        continue 

                    video_id = os.path.splitext(video_file)[0]
                    save_video_to_folder(padded, output_folder, term_name, video_id)
                    total_videos += 1

    print(f"\nDone processing {total_videos} videos into '{output_folder}' folder.")



# Run Augmentation
#process_augmentation()

# Run Resizing and Padding
#process_resize_pad(aug_folder, zero_pad_resized, target_size, target_frames)
