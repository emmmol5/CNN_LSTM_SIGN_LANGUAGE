from IMPORTS import *
metadata = r"C:\Users\Emma\OneDrive - Noroff Education AS\3. Året\Bachelor\Sign Language Health\helse_ordliste_mod.xlsx"
resized_folder = r"C:\Users\Emma\OneDrive - Noroff Education AS\3. Året\Bachelor\Sign Language Health\fixed_resized_frames"
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

def process_videos():
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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB
        frames.append(frame)
    cap.release()
    return np.array(frames)  # Shape: (num_frames, H, W, C)

def uniform_sample_video(video, T=50):
    num_frames = video.shape[0]
    if num_frames < T:
        indices = np.linspace(0, num_frames - 1, num_frames).astype(int)
        extra_indices = np.random.choice(indices, T - num_frames, replace=True)
        indices = np.concatenate([indices, extra_indices])
    else:
        indices = np.linspace(0, num_frames - 1, T).astype(int)
    return video[indices]


'''def resize_frames(video, target_size=(128, 128)):
    return np.array([cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4) for frame in video])


def save_video_to_folder(video, output_path, term_name, video_id):
    term_folder = os.path.join(output_path, term_name)
    os.makedirs(term_folder, exist_ok=True)
    video_path = os.path.join(term_folder, f"{video_id}.mp4")

    video = video.astype(np.uint8)
    height, width, _ = video[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

    for frame in video:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Video saved to {video_path}")



def process_augmented_videos(aug_folder, output_folder, T=80, target_size=(128, 128)):
    for term_name in os.listdir(aug_folder):
        term_folder = os.path.join(aug_folder, term_name)
        if os.path.isdir(term_folder):
            for video_id in os.listdir(term_folder):
                video_path = os.path.join(term_folder, video_id)
                if video_path.endswith('.mp4'):
                    video = load_video(video_path)
                    video_resampled = uniform_sample_video(video, T)
                    video_resized = resize_frames(video_resampled, target_size)
                    save_video_to_folder(video_resized, output_folder, term_name, video_id.split('.')[0])'''

# Run Augmentation and Processing
#process_videos()
#process_augmented_videos(aug_folder, resized_folder, T=80, target_size=(128, 128))

'''# Load and process resized videos for training
def load_videos_from_folder(folder_path, T=80):
    features = []
    labels = []
    for term_name in os.listdir(folder_path):
        term_folder = os.path.join(folder_path, term_name)
        if os.path.isdir(term_folder):
            for video_id in os.listdir(term_folder):
                video_path = os.path.join(term_folder, video_id)
                if video_path.endswith('.mp4'):
                    video = load_video(video_path)
                    video_resampled = uniform_sample_video(video, T)  # Uniformly sample frames
                    features.append(video_resampled)
                    labels.append(term_name)  # The label is the folder name (sign term)
    
    return np.array(features), np.array(labels)

X, y = load_videos_from_folder(zero_pad_resized, T=80)
np.save('processed_features.npy', X)
np.save('processed_labels.npy', y)

print(X.shape)'''




target_size = (128, 128)
T = 80  

def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def resize_frames(video, target_size=(128, 128)):
    return [cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4) for frame in video]


def zero_pad_frames(frames, target_frame_count=80):
    frame_count = len(frames)
    if frame_count > target_frame_count:
        step = max(frame_count // target_frame_count, 1)  
        result = [frames[i] for i in range(0, frame_count, step)]
        # Ensure we do not exceed the target frame count
        if len(result) > target_frame_count:
            result = result[:target_frame_count]
        print(f"[INFO] Downsampling: {frame_count} -> {len(result)} frames")  
        return result
    elif frame_count < target_frame_count:
        padding = [np.zeros_like(frames[0], dtype=np.uint8) for _ in range(target_frame_count - frame_count)]
        result = frames + padding
        print(f"[INFO] Padding: {frame_count} -> {len(result)} frames")  
        return result
    else:
        return frames



def save_video_to_folder(video_frames, output_path, term_name, video_id):
    term_folder = os.path.join(output_path, term_name)
    os.makedirs(term_folder, exist_ok=True)
    save_path = os.path.join(term_folder, f"{video_id}.mp4")

    height, width, _ = video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 10.0, (width, height))  
    for frame in video_frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Saved: {save_path}")


def process_videos(aug_folder, output_folder, target_size=(128, 128), target_frames=80):
    total_videos = 0
    for term_name in os.listdir(aug_folder):
        term_path = os.path.join(aug_folder, term_name)
        if os.path.isdir(term_path):
            for video_file in os.listdir(term_path):
                if video_file.endswith('.mp4'):
                    video_path = os.path.join(term_path, video_file)
                    frames = load_video(video_path)

                    if len(frames) == 0:
                        print(f"[ERROR] Could not read frames from {video_path}")
                        continue

                    resized = resize_frames(frames, target_size)
                    padded = zero_pad_frames(resized, target_frames)

                    if len(padded) != target_frames:
                        print(f"[WARNING] {video_path} ended up with {len(padded)} frames!")
                        continue 

                    video_id = os.path.splitext(video_file)[0]
                    save_video_to_folder(padded, output_folder, term_name, video_id)
                    total_videos += 1

    print(f"\nDone processing {total_videos} videos into '{output_folder}' folder.")

#process_videos(aug_folder, zero_pad_resized, target_size, T)