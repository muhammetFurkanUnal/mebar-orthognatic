import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import cfg


w = cfg.img_size[0]
h = cfg.img_size[1]

train_transform = A.Compose(
    [
        # 1. Geometric adjustments (Mild & Anatomically realistic)
        A.Affine(
            scale=(0.92, 1.08),             # 0.08 scale limit
            translate_percent=(-0.05, 0.05), # 5% shift limit
            rotate=(-10, 10),               # 10 degrees rotation limit
            fill=0,                         # Padding with black border
            p=0.7
        ),
        
        # 2. Photometric / Sensor variations
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5
        ),
        A.RandomGamma(gamma_limit=(85, 115), p=0.3),
        
        # 3. Noise & Blur (Lightweight)
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
        ], p=0.25),
        
        # 4. Final resize and tensor conversion
        A.Resize(height=h, width=w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)

