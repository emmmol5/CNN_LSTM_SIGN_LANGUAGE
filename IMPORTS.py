import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import time
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import shutil
import numpy as np
import pandas as pd
import cv2
import re
import random
import shutil
import vidaug.augmentors as va 
from tqdm import tqdm
import moviepy.editor as mp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


org_metadata = "helse_ordliste.xlsx" # Originale excel filen
metadata = "helse_ordliste_mod.xlsx" # Fikset excel fil (endret navn og fjernet en rad)
video_folder = "helse_tegn" # Originale videoer
aug_folder = "aug_videos" # Kun augmentert 
resized_folder = "fixed_resized_frames" # 128x128, T=50
zero_pad_resized = "vid_zeropad_resized" # Resizet til 128x128, T=80
