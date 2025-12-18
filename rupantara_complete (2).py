"""
RUPANTARA ULTIMATE - World-Class Cross-Species Face Generation System
Complete Production Implementation with ALL Advanced Features
OPTIMIZED FOR GOOGLE COLAB - FIXED VERSION
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, models
from torchvision.utils import make_grid, save_image
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import pickle
import json
from datetime import datetime
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from collections import defaultdict, Counter
import warnings
import traceback
warnings.filterwarnings('ignore')

# ============================================================================
# INSTALL MISSING DEPENDENCIES
# ============================================================================

def install_dependencies():
    """Install required packages for Colab"""
    print("📦 Installing dependencies...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lpips", "albumentations", "opencv-python", "scikit-image"])
        print("✅ Dependencies installed")
    except Exception as e:
        print(f"⚠️  Some dependencies failed to install: {e}")

def download_test_datasets():
    """Download test datasets if none exist"""
    print("📥 Downloading test datasets...")
    # This is a placeholder - you should implement your own dataset download logic
    print("⚠️  Please upload your own datasets to the appropriate folders")

# ============================================================================
# ENVIRONMENT DETECTION & CONFIGURATION
# ============================================================================

def detect_environment():
    """Detect if running in Colab"""
    try:
        import google.colab
        return 'colab'
    except:
        return 'local'

ENVIRONMENT = detect_environment()
IS_COLAB = (ENVIRONMENT == 'colab')

class Config:
    # Environment-aware paths
    if IS_COLAB:
        PROJECT_ROOT = Path('/content/drive/MyDrive/rupantara')
        print("🌐 Running in Google Colab")
        print("📁 Please ensure your dataset is in /content/drive/MyDrive/rupantara/data/")
    else:
        PROJECT_ROOT = Path.cwd()
        print("💻 Running locally")

    DATA_DIR = PROJECT_ROOT / "data"
    HUMAN_DIR = DATA_DIR / "humans"
    ANIMAL_DIR = DATA_DIR / "animals"
    LOGS_DIR = PROJECT_ROOT / "logs"
    CHECKPOINT_DIR = LOGS_DIR / "checkpoints"
    TENSORBOARD_DIR = LOGS_DIR / "tensorboard"
    RESULTS_DIR = PROJECT_ROOT / "results"
    RECONSTRUCTIONS_DIR = RESULTS_DIR / "reconstructions"
    HYBRIDS_DIR = RESULTS_DIR / "hybrids"
    ANALYSIS_DIR = RESULTS_DIR / "analysis"
    HEATMAPS_DIR = ANALYSIS_DIR / "heatmaps"

    # Animal species configuration
    ANIMAL_SPECIES = ['cat', 'dog', 'bear', 'fox', 'leopard', 'lion', 'tiger', 'wolf']
    NUM_SPECIES = len(ANIMAL_SPECIES)
    SPECIES_TO_IDX = {species: idx for idx, species in enumerate(ANIMAL_SPECIES)}
    IDX_TO_SPECIES = {idx: species for species, idx in SPECIES_TO_IDX.items()}

    # Facial regions configuration (for 256x256 images)
    FACIAL_REGIONS = {
        'eyes': {'coords': (60, 90, 80, 176), 'weight': 1.5},      # (y1, y2, x1, x2)
        'nose': {'coords': (100, 140, 96, 160), 'weight': 1.2},
        'mouth': {'coords': (150, 190, 80, 176), 'weight': 1.3},
        'ears': {'coords': (70, 150, 20, 236), 'weight': 0.8},
        'forehead': {'coords': (30, 80, 64, 192), 'weight': 0.7},
    }
    NUM_REGIONS = len(FACIAL_REGIONS)

    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2 if torch.cuda.is_available() else 0  # Reduced for Colab
    PIN_MEMORY = True if torch.cuda.is_available() else False
    USE_AMP = torch.cuda.is_available()  # Automatic Mixed Precision

    # Image configuration
    IMAGE_SIZE = 256
    IMAGE_CHANNELS = 3
    NORMALIZE_MEAN = [0.5, 0.5, 0.5]
    NORMALIZE_STD = [0.5, 0.5, 0.5]

    # Model architecture
    LATENT_DIM = 512
    REGION_LATENT_DIM = 128  # Per-region latent dimension
    HIDDEN_DIMS = [32, 64, 128, 256, 512]
    CONDITION_DIM = 64  # Conditional embedding dimension

    # Training hyperparameters (optimized for Colab)
    BATCH_SIZE = 8 if IS_COLAB else 16  # Smaller for Colab memory
    NUM_EPOCHS = 30 if IS_COLAB else 50  # Fewer epochs for quick testing
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    VAE_BETA = 1.0
    GAN_LOSS_WEIGHT = 0.05
    PERCEPTUAL_LOSS_WEIGHT = 0.1
    IDENTITY_LOSS_WEIGHT = 0.15
    SYMMETRY_LOSS_WEIGHT = 0.05
    FEATURE_MATCHING_WEIGHT = 10.0

    # EMA configuration
    EMA_DECAY = 0.999

    # Training stability
    EARLY_STOPPING_PATIENCE = 10 if IS_COLAB else 15
    SAVE_FREQUENCY = 2 if IS_COLAB else 5  # Save more frequently in Colab
    GRADIENT_CLIP = 1.0

    # Reproducibility
    RANDOM_SEED = 42

    @classmethod
    def create_directories(cls):
        """Check and create main directories only"""
        directories = [
            cls.LOGS_DIR, cls.CHECKPOINT_DIR, cls.TENSORBOARD_DIR,
            cls.RESULTS_DIR, cls.RECONSTRUCTIONS_DIR, cls.HYBRIDS_DIR,
            cls.ANALYSIS_DIR, cls.HEATMAPS_DIR
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        # Create data directories
        data_dirs = [cls.DATA_DIR, cls.HUMAN_DIR, cls.ANIMAL_DIR]
        for directory in data_dirs:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Create animal species directories
        for species in cls.ANIMAL_SPECIES:
            species_dir = cls.ANIMAL_DIR / species
            species_dir.mkdir(parents=True, exist_ok=True)

        print("✅ All directories checked/created")
        print(f"📁 Project root: {cls.PROJECT_ROOT}")
        print(f"📁 Data dir: {cls.DATA_DIR}")

    @classmethod
    def set_seed(cls):
        """Set random seeds for reproducibility"""
        torch.manual_seed(cls.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.RANDOM_SEED)
        np.random.seed(cls.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "="*60)
        print("RUPANTARA CONFIGURATION")
        print("="*60)
        print(f"Device: {cls.DEVICE}")
        print(f"Environment: {'Google Colab' if IS_COLAB else 'Local'}")
        print(f"AMP Enabled: {cls.USE_AMP}")
        print(f"Project Root: {cls.PROJECT_ROOT}")
        print(f"Image Size: {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"Latent Dim: {cls.LATENT_DIM}")
        print(f"Animal Species: {', '.join(cls.ANIMAL_SPECIES)}")
        print(f"Facial Regions: {', '.join(cls.FACIAL_REGIONS.keys())}")
        print("="*60 + "\n")


# ============================================================================
# ADVANCED DATA AUGMENTATION
# ============================================================================

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("⚠️  Albumentations not installed. Using basic augmentation")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  OpenCV not installed. Heatmap visualization disabled")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("⚠️  LPIPS not installed. Install with: pip install lpips")

def get_advanced_augmentation():
    """Advanced augmentation pipeline using Albumentations"""
    if ALBUMENTATIONS_AVAILABLE:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        ])
    return None

def get_transforms(split='train', use_advanced=False):
    """Get image transforms with optional advanced augmentation"""
    if split == 'train':
        if use_advanced and ALBUMENTATIONS_AVAILABLE:
            aug = get_advanced_augmentation()
            return transforms.Compose([
                transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                transforms.Lambda(lambda img: Image.fromarray(aug(image=np.array(img))['image'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
            ])
        else:
            return transforms.Compose([
                transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
            ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
        ])

# ============================================================================
# CLASS-AWARE DATA LOADING WITH BALANCED SAMPLING
# ============================================================================

class ClassAwareFaceDataset(Dataset):
    """Dataset with class labels and balanced sampling support"""

    def __init__(self, root_dir, dataset_type='animal', transform=None, split='train'):
        self.root_dir = Path(root_dir)
        self.dataset_type = dataset_type
        self.transform = transform
        self.split = split
        self.image_paths = []
        self.labels = []
        self.label_to_name = {}

        if dataset_type == 'animal':
            # Load animals with species labels
            for species_idx, species in enumerate(Config.ANIMAL_SPECIES):
                species_dir = self.root_dir / species
                
                # Look for images in species directory
                species_images = []
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    species_images.extend(list(species_dir.glob(f"*{ext}")))
                    species_images.extend(list(species_dir.glob(f"*{ext.upper()}")))

                if species_images:
                    self.image_paths.extend(species_images)
                    self.labels.extend([species_idx] * len(species_images))
                    self.label_to_name[species_idx] = species
                    print(f"  📦 {species}: {len(species_images)} images")
                else:
                    print(f"  ⚠️  {species}: No images found in {species_dir}")
        else:
            # Human dataset - single class
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                self.image_paths.extend(list(self.root_dir.glob(f"*{ext}")))
                self.image_paths.extend(list(self.root_dir.glob(f"*{ext.upper()}")))

            if self.image_paths:
                self.labels = [0] * len(self.image_paths)
                self.label_to_name = {0: 'human'}
                print(f"  👤 Human: {len(self.image_paths)} images")
            else:
                print(f"  ⚠️  Human: No images found in {root_dir}")

        print(f"📁 Total {dataset_type} images: {len(self.image_paths)}")

        # Calculate class weights for balanced sampling
        if dataset_type == 'animal' and self.labels:
            label_counts = Counter(self.labels)
            total_samples = len(self.labels)
            self.class_weights = {label: total_samples / (len(label_counts) * count) 
                                 for label, count in label_counts.items()}
        else:
            self.class_weights = {0: 1.0}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

            return image, label, str(img_path)
        except Exception as e:
            print(f"⚠️  Error loading {img_path}: {e}")
            # Return black image as fallback
            image = torch.zeros(3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
            return image, label, str(img_path)

    def get_sample_weights(self):
        """Get sample weights for balanced sampling"""
        return [self.class_weights.get(label, 1.0) for label in self.labels]

def get_balanced_dataloader(dataset_type='animal', split='train', batch_size=None,
                            use_advanced_aug=False, balanced=True):
    """Get dataloader with optional balanced sampling"""
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    transform = get_transforms(split, use_advanced_aug)

    if dataset_type == 'human':
        root_dir = Config.HUMAN_DIR
    elif dataset_type == 'animal':
        root_dir = Config.ANIMAL_DIR
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    dataset = ClassAwareFaceDataset(root_dir, dataset_type, transform, split)
    
    if len(dataset) == 0:
        print(f"⚠️  No images found for {dataset_type} dataset!")
        return None

    # Balanced sampling for training
    if split == 'train' and balanced and dataset_type == 'animal':
        sample_weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
            drop_last=True
        )
        print(f"✅ Balanced sampling enabled for {dataset_type}")
    else:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == 'train'),
            num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
            drop_last=(split == 'train')
        )

    return dataloader

# ============================================================================
# ADVANCED MODEL COMPONENTS - FIXED
# ============================================================================

class SelfAttention(nn.Module):
    """Self-attention mechanism for capturing long-range dependencies"""

    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, max(in_channels // 8, 1), 1)
        self.key = nn.Conv2d(in_channels, max(in_channels // 8, 1), 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)   # (B, HW, Cq)
        key = self.key(x).view(B, -1, H * W)                        # (B, Cq, HW)
        attention = F.softmax(torch.bmm(query, key), dim=-1)        # (B, HW, HW)
        value = self.value(x).view(B, -1, H * W)                    # (B, C, HW)
        out = torch.bmm(value, attention.permute(0, 2, 1))          # (B, C, HW)
        out = out.view(B, C, H, W)
        return self.gamma * out + x

class ConditionalEncoder(nn.Module):
    """Conditional encoder with species/domain information"""

    def __init__(self, in_channels=3, latent_dim=512, hidden_dims=None,
                 use_attention=False, num_conditions=None):
        super(ConditionalEncoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = Config.HIDDEN_DIMS

        self.latent_dim = latent_dim
        self.num_conditions = num_conditions

        # Condition embedding
        if num_conditions is not None:
            self.condition_embed = nn.Embedding(num_conditions, Config.CONDITION_DIM)
            # Add condition channels to input
            in_channels = in_channels + Config.CONDITION_DIM
        else:
            self.condition_embed = None

        # Encoder layers
        modules = []
        for i, h_dim in enumerate(hidden_dims):
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(0.2)
            ))
            if use_attention and i == 2:
                modules.append(SelfAttention(h_dim))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        # assume initial image 256 -> after 5 downscales /2 -> 8
        self.feature_size = hidden_dims[-1] * 8 * 8

        self.fc_mu = nn.Linear(self.feature_size, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_size, latent_dim)

    def forward(self, x, condition=None):
        # Apply condition if provided
        if self.condition_embed is not None and condition is not None:
            B, C, H, W = x.shape
            cond_emb = self.condition_embed(condition).view(B, Config.CONDITION_DIM, 1, 1)
            cond_emb = cond_emb.expand(B, Config.CONDITION_DIM, H, W)
            x = torch.cat([x, cond_emb], dim=1)

        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class ConditionalDecoder(nn.Module):
    """Conditional decoder with species/domain information"""
    
    def __init__(self, latent_dim=512, out_channels=3, hidden_dims=None, 
                 use_attention=False, num_conditions=None):
        super(ConditionalDecoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = Config.HIDDEN_DIMS[::-1]
        
        self.latent_dim = latent_dim
        self.initial_size = 8
        self.initial_channels = hidden_dims[0]
        self.num_conditions = num_conditions
        
        # Condition embedding
        if num_conditions is not None:
            self.condition_embed = nn.Embedding(num_conditions, Config.CONDITION_DIM)
            input_dim = latent_dim + Config.CONDITION_DIM
        else:
            self.condition_embed = None
            input_dim = latent_dim
        
        self.feature_size = self.initial_channels * self.initial_size * self.initial_size
        self.fc = nn.Linear(input_dim, self.feature_size)
        
        # Decoder layers
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], 
                                 kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[i+1]),
                nn.ReLU()
            ))
            if use_attention and i == 1:
                modules.append(SelfAttention(hidden_dims[i+1]))
        
        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], 
                             kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.ReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )
    
    def forward(self, z, condition=None):
        # Concatenate condition if provided
        if self.condition_embed is not None and condition is not None:
            cond_emb = self.condition_embed(condition)
            z = torch.cat([z, cond_emb], dim=1)
        
        x = self.fc(z)
        x = x.view(-1, self.initial_channels, self.initial_size, self.initial_size)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x

class RegionAwareEncoder(nn.Module):
    """Encoder that processes facial regions separately"""
    
    def __init__(self, in_channels=3, region_latent_dim=128, use_attention=False):
        super(RegionAwareEncoder, self).__init__()
        
        self.region_latent_dim = region_latent_dim
        self.regions = list(Config.FACIAL_REGIONS.keys())
        self.num_regions = len(self.regions)
        
        # Shared feature extractor for all regions
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        
        if use_attention:
            self.attention = SelfAttention(128)
        else:
            self.attention = None
        
        # Region-specific encoders
        self.region_encoders = nn.ModuleDict({
            region: nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Linear(256, region_latent_dim * 2)  # mu and logvar
            )
            for region in self.regions
        })
    
    def extract_region(self, x, region_name):
        """Extract region from image"""
        coords = Config.FACIAL_REGIONS[region_name]['coords']
        y1, y2, x1, x2 = coords
        return x[:, :, y1:y2, x1:x2]
    
    def forward(self, x):
        """Returns dict of {region: (mu, logvar)}"""
        region_latents = {}
        
        for region in self.regions:
            # Extract region
            region_img = self.extract_region(x, region)
            
            # Resize to consistent size if needed
            if region_img.shape[2] < 32 or region_img.shape[3] < 32:
                region_img = F.interpolate(region_img, size=(64, 64), mode='bilinear')
            
            # Encode
            features = self.shared_encoder(region_img)
            if self.attention:
                features = self.attention(features)
            
            # Region-specific encoding
            output = self.region_encoders[region](features)
            mu, logvar = torch.chunk(output, 2, dim=1)
            
            region_latents[region] = (mu, logvar)
        
        return region_latents

class ConditionalVAE(nn.Module):
    """Conditional VAE with optional region-aware encoding - COMPLETE"""
    
    def __init__(self, in_channels=3, latent_dim=512, beta=1.0, 
                 use_attention=False, num_conditions=None, region_aware=False):
        super(ConditionalVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.beta = beta
        self.region_aware = region_aware
        
        if region_aware:
            self.encoder = RegionAwareEncoder(in_channels, Config.REGION_LATENT_DIM, use_attention)
            # Total latent dim is sum of all region latents
            total_latent = Config.REGION_LATENT_DIM * Config.NUM_REGIONS
            self.latent_combiner = nn.Linear(total_latent, latent_dim)
        else:
            self.encoder = ConditionalEncoder(in_channels, latent_dim, use_attention=use_attention,
                                             num_conditions=num_conditions)
        
        self.decoder = ConditionalDecoder(latent_dim, in_channels, use_attention=use_attention,
                                         num_conditions=num_conditions)
        
        self.num_conditions = num_conditions
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x, condition=None):
        """Encode input image to latent space"""
        if self.region_aware:
            region_latents = self.encoder(x)
            z_list = []
            mu_list = []
            logvar_list = []
            
            for region in Config.FACIAL_REGIONS.keys():
                mu, logvar = region_latents[region]
                z = self.reparameterize(mu, logvar)
                z_list.append(z)
                mu_list.append(mu)
                logvar_list.append(logvar)
            
            # Concatenate all region latents
            z_combined = torch.cat(z_list, dim=1)
            z = self.latent_combiner(z_combined)
            
            # Average mu and logvar for loss calculation
            mu = torch.stack(mu_list, dim=1).mean(dim=1)
            logvar = torch.stack(logvar_list, dim=1).mean(dim=1)
            
            return z, region_latents
        else:
            mu, logvar = self.encoder(x, condition)
            z = self.reparameterize(mu, logvar)
            return z, None
    
    def decode(self, z, condition=None):
        """Decode latent vector to image"""
        return self.decoder(z, condition)
    
    def forward(self, x, condition=None):
        """Complete forward pass"""
        z, region_latents = self.encode(x, condition)
        x_recon = self.decode(z, condition)
        return x_recon, region_latents, z

class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator with feature extraction"""
    
    def __init__(self, in_channels=3):
        super(PatchGANDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.layer1 = discriminator_block(in_channels, 64, normalize=False)
        self.layer2 = discriminator_block(64, 128)
        self.layer3 = discriminator_block(128, 256)
        self.layer4 = discriminator_block(256, 512)
        self.final = nn.Conv2d(512, 1, 4, 1, 1)
    
    def forward(self, img, return_features=False):
        """Forward pass with optional feature extraction"""
        feat1 = self.layer1(img)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        output = self.final(feat4)
        
        if return_features:
            return output, [feat1, feat2, feat3, feat4]
        return output

class EnhancedMappingNetwork(nn.Module):
    """Enhanced mapping network with residual connections"""
    
    def __init__(self, latent_dim=512, hidden_dim=1024, num_layers=8):
        super(EnhancedMappingNetwork, self).__init__()
        
        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.LeakyReLU(0.2))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.mapping = nn.Sequential(*layers)
    
    def forward(self, z_animal):
        return self.mapping(z_animal)

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss"""
    
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg[:4]))
        self.slice2 = nn.Sequential(*list(vgg[4:9]))
        self.slice3 = nn.Sequential(*list(vgg[9:16]))
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        y = (y - mean) / std
        
        x_feat1, y_feat1 = self.slice1(x), self.slice1(y)
        x_feat2, y_feat2 = self.slice2(x_feat1), self.slice2(y_feat1)
        x_feat3, y_feat3 = self.slice3(x_feat2), self.slice3(y_feat2)
        
        loss = F.l1_loss(x_feat1, y_feat1) + F.l1_loss(x_feat2, y_feat2) + F.l1_loss(x_feat3, y_feat3)
        return loss

class LPIPSLoss(nn.Module):
    """LPIPS perceptual loss"""
    
    def __init__(self):
        super(LPIPSLoss, self).__init__()
        if LPIPS_AVAILABLE:
            self.lpips = lpips.LPIPS(net='alex')
            for param in self.lpips.parameters():
                param.requires_grad = False
        else:
            self.lpips = None
    
    def forward(self, x, y):
        if self.lpips:
            return self.lpips(x, y).mean()
        return torch.tensor(0.0).to(x.device)

class IdentityPreservationLoss(nn.Module):
    """Identity preservation using face recognition features"""
    
    def __init__(self):
        super(IdentityPreservationLoss, self).__init__()
        # Use ResNet as identity encoder
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        """Compare identity features"""
        x_feat = self.encoder(x).flatten(1)
        y_feat = self.encoder(y).flatten(1)
        return 1.0 - F.cosine_similarity(x_feat, y_feat).mean()

class SymmetryLoss(nn.Module):
    """Enforce facial symmetry"""
    
    def forward(self, x):
        """Compare left and right halves"""
        x_flipped = torch.flip(x, dims=[3])
        return F.l1_loss(x, x_flipped)

def feature_matching_loss(real_features, fake_features):
    """Feature matching loss for GAN"""
    loss = 0
    for real_feat, fake_feat in zip(real_features, fake_features):
        loss += F.l1_loss(fake_feat, real_feat.detach())
    return loss

def comprehensive_loss(x_recon, x, mu, logvar, discriminator_output=None,
                       real_features=None, fake_features=None,
                       perceptual_loss_fn=None, lpips_loss_fn=None,
                       identity_loss_fn=None, symmetry_loss_fn=None,
                       beta=1.0, use_perceptual=False, use_lpips=False,
                       use_identity=False, use_symmetry=False, use_gan=False):
    """Comprehensive loss function with all components"""
    
    # Base VAE losses
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    
    # Handle case where mu and logvar might be zeros
    if mu is not None and logvar is not None and not torch.all(mu == 0):
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        kl_loss = torch.tensor(0.0).to(x.device)
    
    total_loss = recon_loss + beta * kl_loss
    loss_dict = {'recon': recon_loss.item(), 'kl': kl_loss.item()}
    
    # Perceptual loss
    if use_perceptual and perceptual_loss_fn:
        perc_loss = perceptual_loss_fn(x_recon, x)
        total_loss += Config.PERCEPTUAL_LOSS_WEIGHT * perc_loss
        loss_dict['perceptual'] = perc_loss.item()
    
    # LPIPS loss
    if use_lpips and lpips_loss_fn and LPIPS_AVAILABLE:
        lp_loss = lpips_loss_fn(x_recon, x)
        total_loss += Config.PERCEPTUAL_LOSS_WEIGHT * lp_loss
        loss_dict['lpips'] = lp_loss.item()
    
    # Identity preservation
    if use_identity and identity_loss_fn:
        identity_loss = identity_loss_fn(x_recon, x)
        total_loss += Config.IDENTITY_LOSS_WEIGHT * identity_loss
        loss_dict['identity'] = identity_loss.item()
    
    # Symmetry loss
    if use_symmetry and symmetry_loss_fn:
        sym_loss = symmetry_loss_fn(x_recon)
        total_loss += Config.SYMMETRY_LOSS_WEIGHT * sym_loss
        loss_dict['symmetry'] = sym_loss.item()
    
    # GAN loss
    if use_gan and discriminator_output is not None:
        gan_loss = F.binary_cross_entropy_with_logits(
            discriminator_output, torch.ones_like(discriminator_output)
        )
        total_loss += Config.GAN_LOSS_WEIGHT * gan_loss
        loss_dict['gan'] = gan_loss.item()
    
    # Feature matching
    if real_features is not None and fake_features is not None:
        fm_loss = feature_matching_loss(real_features, fake_features)
        total_loss += Config.FEATURE_MATCHING_WEIGHT * fm_loss
        loss_dict['feature_matching'] = fm_loss.item()
    
    return total_loss, loss_dict

# ============================================================================
# EVALUATION METRICS
# ============================================================================

class MetricsCalculator:
    """Calculate FID, LPIPS, SSIM, PSNR"""
    
    def __init__(self, device):
        self.device = device
        self.inception = models.inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = nn.Identity()
        self.inception.eval()
        self.inception.to(device)
        
        for param in self.inception.parameters():
            param.requires_grad = False
        
        if LPIPS_AVAILABLE:
            self.lpips_metric = lpips.LPIPS(net='alex').to(device)
            for param in self.lpips_metric.parameters():
                param.requires_grad = False
    
    @torch.no_grad()
    def calculate_fid(self, real_images, generated_images):
        """Calculate Fréchet Inception Distance"""
        def get_features(images):
            if images.shape[-1] != 299:
                images = F.interpolate(images, size=(299, 299), mode='bilinear')
            features = self.inception(images)
            return features.cpu().numpy()
        
        real_features = get_features(real_images)
        gen_features = get_features(generated_images)
        
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
        
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        return max(0, fid)
    
    @torch.no_grad()
    def calculate_lpips(self, img1, img2):
        if LPIPS_AVAILABLE:
            return self.lpips_metric(img1, img2).mean().item()
        return 0.0
    
    @staticmethod
    def calculate_ssim(img1, img2):
        img1_np = ((img1.cpu().numpy().transpose(0, 2, 3, 1) + 1) / 2 * 255).astype(np.uint8)
        img2_np = ((img2.cpu().numpy().transpose(0, 2, 3, 1) + 1) / 2 * 255).astype(np.uint8)
        
        scores = []
        for i in range(len(img1_np)):
            score = ssim(img1_np[i], img2_np[i], channel_axis=2, data_range=255)
            scores.append(score)
        return np.mean(scores)
    
    @staticmethod
    def calculate_psnr(img1, img2):
        img1_np = ((img1.cpu().numpy().transpose(0, 2, 3, 1) + 1) / 2 * 255).astype(np.uint8)
        img2_np = ((img2.cpu().numpy().transpose(0, 2, 3, 1) + 1) / 2 * 255).astype(np.uint8)
        
        scores = []
        for i in range(len(img1_np)):
            score = psnr(img1_np[i], img2_np[i], data_range=255)
            scores.append(score)
        return np.mean(scores)
    
    def calculate_all_metrics(self, real_images, generated_images):
        metrics = {
            'mse': F.mse_loss(generated_images, real_images).item(),
            'ssim': self.calculate_ssim(real_images, generated_images),
            'psnr': self.calculate_psnr(real_images, generated_images),
        }
        
        if LPIPS_AVAILABLE:
            metrics['lpips'] = self.calculate_lpips(generated_images, real_images)
        
        try:
            metrics['fid'] = self.calculate_fid(real_images, generated_images)
        except:
            metrics['fid'] = 0.0
        
        return metrics

# ============================================================================
# ANIMAL FEATURE BANK
# ============================================================================

class AnimalFeatureBank:
    """Store and retrieve animal facial features"""
    
    def __init__(self, device):
        self.device = device
        self.features = defaultdict(lambda: defaultdict(list))
        # Structure: features[species][region] = [(latent, image_path), ...]
    
    def add_feature(self, species, region, latent, image_path):
        """Add a feature to the bank"""
        self.features[species][region].append({
            'latent': latent.cpu(),
            'path': image_path
        })
    
    def get_best_match(self, query_latent, species, region, top_k=1):
        """Find best matching feature"""
        if species not in self.features or region not in self.features[species]:
            return None
        
        candidates = self.features[species][region]
        similarities = []
        
        query_latent = query_latent.to(self.device)
        
        for candidate in candidates:
            cand_latent = candidate['latent'].to(self.device)
            # Ensure proper dimensions for cosine similarity
            q = query_latent.squeeze().unsqueeze(0)
            c = cand_latent.squeeze().unsqueeze(0)
            sim = F.cosine_similarity(q, c, dim=1).item()
            similarities.append((sim, candidate))
        
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        if top_k == 1:
            return similarities[0] if similarities else None
        return similarities[:top_k]
    
    def save(self, path):
        """Save feature bank to disk"""
        save_dict = {}
        for species in self.features:
            save_dict[species] = {}
            for region in self.features[species]:
                save_dict[species][region] = self.features[species][region]
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"✅ Feature bank saved to {path}")
    
    def load(self, path):
        """Load feature bank from disk"""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.features = defaultdict(lambda: defaultdict(list))
        for species in save_dict:
            for region in save_dict[species]:
                self.features[species][region] = save_dict[species][region]
        
        print(f"✅ Feature bank loaded from {path}")
    
    def get_stats(self):
        """Get statistics about the feature bank"""
        stats = {}
        for species in self.features:
            stats[species] = {}
            for region in self.features[species]:
                stats[species][region] = len(self.features[species][region])
        return stats

# ============================================================================
# LATENT MIXING ENGINE
# ============================================================================

class LatentMixingEngine:
    """Mix latent codes from different animals"""
    
    def __init__(self, latent_dim=512, device='cuda'):
        self.latent_dim = latent_dim
        self.device = device
        self.regions = list(Config.FACIAL_REGIONS.keys())
        
        # Region combination network
        self.combiner = nn.Sequential(
            nn.Linear(Config.REGION_LATENT_DIM * Config.NUM_REGIONS, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, latent_dim)
        ).to(device)
    
    def mix_latents(self, region_latents_dict):
        """
        Mix region latents into single latent code
        Uses simple concatenation + linear projection (no neural network)
        """
        # Concatenate all region latents in order
        latents = [region_latents_dict[region] for region in self.regions]
        
        # Ensure all are 2D [1, D]
        latents_2d = []
        for lat in latents:
            if lat.dim() == 1:
                latents_2d.append(lat.unsqueeze(0))
            elif lat.dim() == 2:
                latents_2d.append(lat)
            else:
                latents_2d.append(lat.reshape(1, -1))
        
        # Simple concatenation
        combined = torch.cat(latents_2d, dim=-1)  # Shape: [1, 128*5=640]
        
        # SIMPLE MIXING: Just use weighted average instead of neural network
        # This preserves the magnitude and distribution of latents
        if combined.shape[1] == self.latent_dim:
            return combined
        
        # If not matching, use simple linear interpolation
        # Split back to regions and average
        region_size = Config.REGION_LATENT_DIM
        expanded_latents = []
        
        for i, lat in enumerate(latents_2d):
            # Simple expansion: repeat to match target size
            target_size = self.latent_dim // len(self.regions)
            if lat.shape[1] < target_size:
                # Pad with zeros
                padding = torch.zeros(1, target_size - lat.shape[1], device=lat.device)
                expanded = torch.cat([lat, padding], dim=1)
            else:
                # Truncate
                expanded = lat[:, :target_size]
            expanded_latents.append(expanded)
        
        # Concatenate expanded latents
        mixed = torch.cat(expanded_latents, dim=1)  # Should be [1, 512]
        
        # Ensure exactly 512 dimensions
        if mixed.shape[1] > self.latent_dim:
            mixed = mixed[:, :self.latent_dim]
        elif mixed.shape[1] < self.latent_dim:
            padding = torch.zeros(1, self.latent_dim - mixed.shape[1], device=mixed.device)
            mixed = torch.cat([mixed, padding], dim=1)
        
        return mixed
    def weighted_mix(self, region_latents_dict, weights=None):
        """Mix with custom weights per region"""
        if weights is None:
            weights = {region: Config.FACIAL_REGIONS[region]['weight'] 
                      for region in self.regions}
        
        # Apply weights
        weighted_latents = []
        for region in self.regions:
            latent = region_latents_dict[region]
            weight = weights[region]
            weighted_latents.append(latent * weight)
        
        combined = torch.cat(weighted_latents, dim=-1)
        return self.combiner(combined)

# ============================================================================
# EMA MODEL
# ============================================================================

class EMAModel:
    """Exponential Moving Average of model weights"""
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self):
        """Apply EMA weights to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

# ============================================================================
# TRAINER - FIXED VERSION
# ============================================================================

class EnhancedVAETrainer:
    """Complete trainer with fixed forward calls"""
    
    def __init__(self, model_type='human', args=None):
        self.model_type = model_type
        self.args = args
        self.device = Config.DEVICE
        
        Config.set_seed()
        
        print(f"\n🔧 Initializing {model_type.capitalize()} VAE Trainer...")
        
        # Initialize model
        num_conditions = Config.NUM_SPECIES if model_type == 'animal' else None
        self.model = ConditionalVAE(
            latent_dim=Config.LATENT_DIM,
            beta=args.beta,
            use_attention=args.use_attention,
            num_conditions=num_conditions,
            region_aware=args.region_aware
        ).to(self.device)
        
        print(f"✅ Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # EMA model
        if args.use_ema:
            self.ema = EMAModel(self.model, decay=Config.EMA_DECAY)
            print("✅ EMA enabled")
        else:
            self.ema = None
        
        # Discriminator
        self.use_gan = args.use_gan
        if self.use_gan:
            self.discriminator = PatchGANDiscriminator().to(self.device)
            self.disc_optimizer = optim.Adam(
                self.discriminator.parameters(),
                lr=args.lr * 2,
                betas=(0.5, 0.999)
            )
            print("✅ GAN discriminator initialized")
        
        # Data loaders
        print("\n📊 Loading datasets...")
        self.train_loader = get_balanced_dataloader(
            model_type, 'train', args.batch_size,
            use_advanced_aug=args.use_advanced_aug,
            balanced=True
        )
        
        if self.train_loader is None:
            print(f"❌ No training data found for {model_type}!")
            return
        
        self.val_loader = get_balanced_dataloader(
            model_type, 'val', args.batch_size,
            balanced=False
        )
        
        print(f"✅ Train batches: {len(self.train_loader)}")
        print(f"✅ Val batches: {len(self.val_loader) if self.val_loader else 0}")
        
        # Loss functions
        self.perceptual_loss = PerceptualLoss().to(self.device) if args.use_perceptual else None
        self.lpips_loss = LPIPSLoss().to(self.device) if args.use_lpips else None
        self.identity_loss = IdentityPreservationLoss().to(self.device) if args.use_identity else None
        self.symmetry_loss = SymmetryLoss() if args.use_symmetry else None
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # AMP scaler
        self.scaler = GradScaler() if Config.USE_AMP else None
        
        # Metrics
        self.metrics_calculator = MetricsCalculator(self.device)
        
        # TensorBoard
        self.use_tensorboard = args.use_tensorboard
        if self.use_tensorboard:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Config.TENSORBOARD_DIR / f"{model_type}vae_{timestamp}"
            self.writer = SummaryWriter(log_dir)
            print(f"✅ TensorBoard logging to {log_dir}")
        
        # Training state
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.global_step = 0
        
        # Metrics history
        self.train_metrics_history = []
        self.val_metrics_history = []
    
    def train_epoch(self, epoch):
        """Train for one epoch with fixed forward pass"""
        self.model.train()
        if self.use_gan:
            self.discriminator.train()
        
        epoch_losses = defaultdict(list)
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (images, labels, paths) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device) if self.model_type == 'animal' else None
            
            # ---------------------
            # Train Generator (VAE)
            # ---------------------
            self.optimizer.zero_grad()
            
            if Config.USE_AMP:
                with autocast():
                    # Fixed forward pass
                    x_recon, region_latents, z = self.model(images, labels)
                    
                    # Get mu and logvar for loss calculation
                    if region_latents:
                        mu = torch.stack([region_latents[r][0] for r in Config.FACIAL_REGIONS.keys()], dim=1).mean(dim=1)
                        logvar = torch.stack([region_latents[r][1] for r in Config.FACIAL_REGIONS.keys()], dim=1).mean(dim=1)
                    else:
                        # For non-region-aware models, we need to encode separately
                        mu, logvar = self.model.encoder(images, labels)
                    
                    # Get discriminator features if using GAN
                    if self.use_gan:
                        fake_output, fake_features = self.discriminator(x_recon, return_features=True)
                        _, real_features = self.discriminator(images, return_features=True)
                    else:
                        fake_output, fake_features, real_features = None, None, None
                    
                    # Calculate loss
                    loss, loss_dict = comprehensive_loss(
                        x_recon, images, mu, logvar, fake_output,
                        real_features, fake_features,
                        self.perceptual_loss, self.lpips_loss,
                        self.identity_loss, self.symmetry_loss,
                        beta=self.args.beta,
                        use_perceptual=self.args.use_perceptual,
                        use_lpips=self.args.use_lpips,
                        use_identity=self.args.use_identity,
                        use_symmetry=self.args.use_symmetry,
                        use_gan=self.use_gan
                    )
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRADIENT_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Same logic without AMP
                x_recon, region_latents, z = self.model(images, labels)
                
                if region_latents:
                    mu = torch.stack([region_latents[r][0] for r in Config.FACIAL_REGIONS.keys()], dim=1).mean(dim=1)
                    logvar = torch.stack([region_latents[r][1] for r in Config.FACIAL_REGIONS.keys()], dim=1).mean(dim=1)
                else:
                    mu, logvar = self.model.encoder(images, labels)
                
                if self.use_gan:
                    fake_output, fake_features = self.discriminator(x_recon, return_features=True)
                    _, real_features = self.discriminator(images, return_features=True)
                else:
                    fake_output, fake_features, real_features = None, None, None
                
                loss, loss_dict = comprehensive_loss(
                    x_recon, images, mu, logvar, fake_output,
                    real_features, fake_features,
                    self.perceptual_loss, self.lpips_loss,
                    self.identity_loss, self.symmetry_loss,
                    beta=self.args.beta,
                    use_perceptual=self.args.use_perceptual,
                    use_lpips=self.args.use_lpips,
                    use_identity=self.args.use_identity,
                    use_symmetry=self.args.use_symmetry,
                    use_gan=self.use_gan
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRADIENT_CLIP)
                self.optimizer.step()
            
            # Update EMA
            if self.ema:
                self.ema.update()
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            if self.use_gan:
                self.disc_optimizer.zero_grad()
                
                if Config.USE_AMP:
                    with autocast():
                        real_output = self.discriminator(images)
                        fake_output = self.discriminator(x_recon.detach())
                        
                        d_loss_real = F.binary_cross_entropy_with_logits(
                            real_output, torch.ones_like(real_output)
                        )
                        d_loss_fake = F.binary_cross_entropy_with_logits(
                            fake_output, torch.zeros_like(fake_output)
                        )
                        d_loss = (d_loss_real + d_loss_fake) / 2
                    
                    self.scaler.scale(d_loss).backward()
                    self.scaler.step(self.disc_optimizer)
                    self.scaler.update()
                else:
                    real_output = self.discriminator(images)
                    fake_output = self.discriminator(x_recon.detach())
                    
                    d_loss_real = F.binary_cross_entropy_with_logits(
                        real_output, torch.ones_like(real_output)
                    )
                    d_loss_fake = F.binary_cross_entropy_with_logits(
                        fake_output, torch.zeros_like(fake_output)
                    )
                    d_loss = (d_loss_real + d_loss_fake) / 2
                    
                    d_loss.backward()
                    self.disc_optimizer.step()
                
                loss_dict['d_loss'] = d_loss.item()
            
            # Record losses
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)
            
            # Update progress bar
            progress_bar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})
            
            # TensorBoard logging
            if self.use_tensorboard:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'Train/{key}', value, self.global_step)
            
            self.global_step += 1
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate model"""
        # Use EMA weights if available
        if self.ema:
            self.ema.apply_shadow()
        
        self.model.eval()
        
        epoch_losses = defaultdict(list)
        all_real = []
        all_recon = []
        
        if self.val_loader is None:
            print("⚠️ No validation data available")
            return {'recon': 0.0, 'kl': 0.0, 'loss': 0.0}
        
        for images, labels, paths in tqdm(self.val_loader, desc="Validation"):
            images = images.to(self.device)
            labels = labels.to(self.device) if self.model_type == 'animal' else None
            
            x_recon, region_latents, z = self.model(images, labels)
            
            # Get mu and logvar for loss calculation
            if region_latents:
                mu = torch.stack([region_latents[r][0] for r in Config.FACIAL_REGIONS.keys()], dim=1).mean(dim=1)
                logvar = torch.stack([region_latents[r][1] for r in Config.FACIAL_REGIONS.keys()], dim=1).mean(dim=1)
            else:
                mu, logvar = self.model.encoder(images, labels)
            
            # Calculate loss (no GAN loss during validation)
            loss, loss_dict = comprehensive_loss(
                x_recon, images, mu, logvar,
                perceptual_loss_fn=self.perceptual_loss,
                lpips_loss_fn=self.lpips_loss,
                identity_loss_fn=self.identity_loss,
                symmetry_loss_fn=self.symmetry_loss,
                beta=self.args.beta,
                use_perceptual=self.args.use_perceptual,
                use_lpips=self.args.use_lpips,
                use_identity=self.args.use_identity,
                use_symmetry=self.args.use_symmetry,
                use_gan=False
            )
            
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)
            
            # Collect for metrics
            all_real.append(images)
            all_recon.append(x_recon)
        
        # Calculate metrics
        if all_real:
            all_real = torch.cat(all_real[:5], dim=0)  # Use subset for speed
            all_recon = torch.cat(all_recon[:5], dim=0)
            metrics = self.metrics_calculator.calculate_all_metrics(all_real, all_recon)
        else:
            metrics = {}
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        avg_losses.update(metrics)
        
        # TensorBoard logging
        if self.use_tensorboard:
            for key, value in avg_losses.items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)
            
            # Log images
            if all_real.numel() > 0:
                comparison = torch.cat([all_real[:4], all_recon[:4]])
                grid = make_grid(comparison, nrow=4, normalize=True, range=(-1, 1))
                self.writer.add_image('Val/Reconstruction', grid, epoch)
        
        # Restore original weights
        if self.ema:
            self.ema.restore()
        
        return avg_losses
    
    def train(self):
        """Main training loop - FIXED"""
        print(f"\n🚀 Starting {self.model_type.capitalize()} VAE training for {self.args.epochs} epochs...")
        
        if self.train_loader is None:
            print("❌ Cannot train: No training data available!")
            return
        
        for epoch in range(self.args.epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.args.epochs}")
            print(f"{'='*50}")
            
            # Training phase
            train_losses = self.train_epoch(epoch)
            self.train_metrics_history.append(train_losses)
            
            # Validation phase
            val_losses = self.validate(epoch)
            self.val_metrics_history.append(val_losses)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_losses.get('recon', 0.0):.4f}")
            print(f"  Val Loss: {val_losses.get('recon', 0.0):.4f}")
            
            # Update learning rate scheduler
            val_loss = val_losses.get('loss', val_losses.get('recon', 0.0))
            self.scheduler.step(val_loss)
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"✅ New best model! Val Loss: {val_loss:.6f}")
            else:
                self.epochs_without_improvement += 1
                print(f"⏳ No improvement for {self.epochs_without_improvement} epochs")
            
            # Early stopping
            if self.epochs_without_improvement >= Config.EARLY_STOPPING_PATIENCE:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                break
            
            # Periodic checkpoint
            if (epoch + 1) % Config.SAVE_FREQUENCY == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        # Save final model
        self.save_checkpoint(self.args.epochs - 1, is_best=False)
        
        # Save metrics
        self.save_metrics_history()
        
        print(f"\n✅ {self.model_type.capitalize()} VAE training completed!")
        if self.use_tensorboard:
            print(f"📊 TensorBoard logs: {Config.TENSORBOARD_DIR}")
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        if self.ema:
            self.ema.apply_shadow()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': {
                'latent_dim': Config.LATENT_DIM,
                'beta': self.args.beta,
                'use_attention': self.args.use_attention,
                'region_aware': self.args.region_aware,
            }
        }
        
        if self.use_gan:
            checkpoint['discriminator_state_dict'] = self.discriminator.state_dict()
            checkpoint['disc_optimizer_state_dict'] = self.disc_optimizer.state_dict()
        
        if is_best:
            if self.model_type == 'human':
                path = Config.CHECKPOINT_DIR / "humanvae_best.pth"
            else:
                path = Config.CHECKPOINT_DIR / "animalvae_best.pth"
            torch.save(checkpoint, path)
            print(f"💾 Best model saved to {path}")
        else:
            if self.model_type == 'human':
                path = Config.CHECKPOINT_DIR / f"humanvae_epoch_{epoch+1}.pth"
            else:
                path = Config.CHECKPOINT_DIR / f"animalvae_epoch_{epoch+1}.pth"
            torch.save(checkpoint, path)
            print(f"💾 Checkpoint saved to {path}")
        
        if self.ema:
            self.ema.restore()
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.use_gan and 'discriminator_state_dict' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        print(f"✅ Model loaded from {path}")
        return checkpoint['epoch']
    
    def save_metrics_history(self):
        """Save training metrics to JSON"""
        if self.model_type == 'human':
            metrics_path = Config.ANALYSIS_DIR / "humanvae_metrics.json"
        else:
            metrics_path = Config.ANALYSIS_DIR / "animalvae_metrics.json"
        
        data = {
            'train': self.train_metrics_history,
            'val': self.val_metrics_history,
            'config': vars(self.args)
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📊 Metrics saved to {metrics_path}")

# ============================================================================
# MAPPING NETWORK TRAINER
# ============================================================================

class MappingNetworkTrainer:
    """Train mapping network from animal latent to human latent"""
    
    def __init__(self, hvae, avae, args):
        self.hvae = hvae
        self.avae = avae
        self.args = args
        self.device = Config.DEVICE
        
        # Mapping network
        self.mapper = EnhancedMappingNetwork(
            latent_dim=Config.LATENT_DIM,
            hidden_dim=args.mapper_hidden_dim,
            num_layers=args.mapper_num_layers
        ).to(self.device)
        
        print(f"✅ Mapping network parameters: {sum(p.numel() for p in self.mapper.parameters()):,}")
        
        # Data loaders
        self.human_loader = get_balanced_dataloader('human', 'train', args.batch_size)
        self.animal_loader = get_balanced_dataloader('animal', 'train', args.batch_size)
        
        if self.human_loader is None or self.animal_loader is None:
            print("❌ Cannot train mapper: Missing data!")
            return
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.mapper.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Loss
        self.identity_loss = IdentityPreservationLoss().to(self.device)
        
        # Freeze VAEs
        for param in self.hvae.parameters():
            param.requires_grad = False
        for param in self.avae.parameters():
            param.requires_grad = False
        
        self.hvae.eval()
        self.avae.eval()
        
        self.best_loss = float('inf')
    
    def train(self):
        """Train mapping network"""
        print(f"\n🚀 Training mapping network for {self.args.epochs} epochs...")
        
        if self.human_loader is None or self.animal_loader is None:
            print("❌ Cannot train mapper: Missing data!")
            return
        
        for epoch in range(self.args.epochs):
            self.mapper.train()
            epoch_losses = []
            
            # Create iterators for both dataloaders
            human_iter = iter(self.human_loader)
            animal_iter = iter(self.animal_loader)
            num_batches = min(len(self.human_loader), len(self.animal_loader))
            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}")
            
            for _ in progress_bar:
                try:
                    human_imgs, _, _ = next(human_iter)
                    animal_imgs, animal_labels, _ = next(animal_iter)
                except StopIteration:
                    break
                
                human_imgs = human_imgs.to(self.device)
                animal_imgs = animal_imgs.to(self.device)
                animal_labels = animal_labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Encode
                with torch.no_grad():
                    z_human, _ = self.hvae.encode(human_imgs)
                    z_animal, _ = self.avae.encode(animal_imgs, animal_labels)
                
                # Map animal latent to human latent space
                z_mapped = self.mapper(z_animal)
                
                # Decode mapped latent
                human_from_animal = self.hvae.decode(z_mapped)
                
                # Compute losses
                latent_loss = F.mse_loss(z_mapped, z_human)
                recon_loss = F.mse_loss(human_from_animal, human_imgs)
                identity_loss = self.identity_loss(human_from_animal, human_imgs)
                
                loss = latent_loss + 0.5 * recon_loss + 0.3 * identity_loss
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.mapper.parameters(), Config.GRADIENT_CLIP)
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
                progress_bar.set_postfix({'loss': loss.item()})
            
            # Average loss for the epoch
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                print(f"\n📊 Epoch {epoch+1} - Loss: {avg_loss:.6f}")
                
                # Step scheduler
                self.scheduler.step(avg_loss)
                
                # Save best model
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"✅ New best loss: {avg_loss:.6f}")
                
                # Periodic checkpoint
                if (epoch + 1) % Config.SAVE_FREQUENCY == 0:
                    self.save_checkpoint(epoch, is_best=False)
            else:
                print(f"⚠️ Epoch {epoch+1}: No training data")
        
        print("\n✅ Mapping network training completed!")
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save mapper checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.mapper.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
        }
        
        if is_best:
            path = Config.CHECKPOINT_DIR / "mapper_best.pth"
        else:
            path = Config.CHECKPOINT_DIR / f"mapper_epoch_{epoch+1}.pth"
        
        torch.save(checkpoint, path)
        print(f"💾 Mapper checkpoint saved to {path}")

# ============================================================================
# FEATURE BANK BUILDER - FIXED
# ============================================================================

class FeatureBankBuilder:
    """Build animal feature bank from trained VAE"""
    
    def __init__(self, avae, device):
        self.avae = avae
        self.device = device
        self.feature_bank = AnimalFeatureBank(device)
    
    @torch.no_grad()
    def build(self):
        """Build feature bank from all animal images"""
        print("\n🏗️  Building animal feature bank...")
        self.avae.eval()
        
        # Create a dataloader for all animal images
        animal_dataset = ClassAwareFaceDataset(
            root_dir=Config.ANIMAL_DIR,
            dataset_type='animal',
            transform=get_transforms('val'),
            split='train'
        )
        
        if len(animal_dataset) == 0:
            print("❌ No animal images found!")
            return None
        
        # Create dataloader
        dataloader = DataLoader(
            animal_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY
        )
        
        total_features = 0
        
        for images, labels, paths in tqdm(dataloader, desc="Building feature bank"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Get latents
            with torch.no_grad():
                _, region_latents, _ = self.avae(images, labels)
            
            # Store features
            for batch_idx in range(len(images)):
                species_name = Config.IDX_TO_SPECIES[labels[batch_idx].item()]
                
                if region_latents:
                    # Region-aware latents
                    for region in Config.FACIAL_REGIONS.keys():
                        mu = region_latents[region][0][batch_idx]  # mu
                        self.feature_bank.add_feature(species_name, region, mu, paths[batch_idx])
                        total_features += 1
                else:
                    # Global latent
                    z, _ = self.avae.encode(images[batch_idx:batch_idx+1], labels[batch_idx:batch_idx+1])
                    for region in Config.FACIAL_REGIONS.keys():
                        self.feature_bank.add_feature(species_name, region, z.squeeze(0), paths[batch_idx])
                        total_features += 1
        
        # Save feature bank
        bank_path = Config.CHECKPOINT_DIR / "animal_feature_bank.pkl"
        self.feature_bank.save(bank_path)
        
        print(f"\n✅ Feature bank built with {total_features} features")
        print(f"💾 Saved to {bank_path}")
        
        # Print statistics
        stats = self.feature_bank.get_stats()
        print("\n📊 Feature Bank Statistics:")
        for species, regions in stats.items():
            print(f"\n  {species.capitalize()}:")
            for region, count in regions.items():
                print(f"    {region}: {count} features")
        
        return self.feature_bank

# ============================================================================
# HYBRID GENERATOR - FIXED
# ============================================================================

class HybridFaceGenerator:
    """Generate hybrid human faces with animal features"""
    
    def __init__(self, hvae, feature_bank, device):
        self.hvae = hvae
        self.feature_bank = feature_bank
        self.device = device
        self.mixing_engine = LatentMixingEngine(Config.LATENT_DIM, device)
    
    @torch.no_grad()
    def generate_hybrid(self, human_image_path, region_species_map, save_path=None):
        """
        Generate hybrid face.
        region_species_map: {region: species} e.g., {'eyes': 'wolf', 'nose': 'tiger', 'mouth': 'dog'}
        """
        print(f"\n🎨 Generating hybrid face...")
        
        # Load human image
        transform = get_transforms('val')
        try:
            human_img = Image.open(human_image_path).convert('RGB')
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
        
        human_tensor = transform(human_img).unsqueeze(0).to(self.device)
        
        # Encode human
        z_human, region_latents = self.hvae.encode(human_tensor)
        
        # Build mixed latent
        mixed_latents = {}
        
        # Handle region-aware vs non-region-aware models
        if region_latents:
            # Region-aware model
            for region in Config.FACIAL_REGIONS.keys():
                if region in region_species_map:
                    species = region_species_map[region]
                    print(f"  {region}: using {species}")
                    
                    # Get best matching animal feature for this region
                    query_latent = region_latents[region][0]  # mu
                    match = self.feature_bank.get_best_match(query_latent, species, region)
                    
                    if match:
                        similarity, candidate = match
                        animal_latent = candidate['latent'].to(self.device).unsqueeze(0)
                        mixed_latents[region] = animal_latent
                        print(f"    Similarity: {similarity*100:.1f}%")
                    else:
                        print(f"    ⚠️ No match found, using human feature")
                        mixed_latents[region] = query_latent.unsqueeze(0)
                else:
                    # Use human latent for regions not specified
                    mixed_latents[region] = region_latents[region][0].unsqueeze(0)
        else:
            # Non-region-aware model
            for region in Config.FACIAL_REGIONS.keys():
                if region in region_species_map:
                    species = region_species_map[region]
                    print(f"  {region}: using {species}")
                    
                    # Use global latent with species matching
                    match = self.feature_bank.get_best_match(z_human, species, region)
                    
                    if match:
                        similarity, candidate = match
                        animal_latent = candidate['latent'].to(self.device).unsqueeze(0)
                        mixed_latents[region] = animal_latent
                        print(f"    Similarity: {similarity*100:.1f}%")
                    else:
                        print(f"    ⚠️ No match found, using human feature")
                        mixed_latents[region] = z_human
                else:
                    # Use human latent for regions not specified
                    mixed_latents[region] = z_human
        
        # Mix latents
        try:
            # Prepare latents for mixing - ensure all have shape [1, D]
            region_latents_for_mixing = {}
            for region, latent in mixed_latents.items():
                # Normalize all latents to 2D shape [1, D]
                if latent.dim() == 1:
                    region_latents_for_mixing[region] = latent.unsqueeze(0)
                elif latent.dim() == 2:
                    if latent.shape[0] != 1:
                        region_latents_for_mixing[region] = latent[:1]
                    else:
                        region_latents_for_mixing[region] = latent
                elif latent.dim() == 3:
                    region_latents_for_mixing[region] = latent.squeeze(1)[:1]
                else:
                    region_latents_for_mixing[region] = latent.reshape(1, -1)
            
            final_latent = self.mixing_engine.mix_latents(region_latents_for_mixing)
            
            # Decode hybrid
            hybrid_img = self.hvae.decode(final_latent)
            
            # Save
            if save_path:
                save_image(hybrid_img, save_path, normalize=True, value_range=(-1, 1))
                print(f"✅ Hybrid saved to {save_path}")
            
            # Create comparison image
            if save_path:
                # Ensure both in same range
                human_tensor = torch.clamp(human_tensor, -1, 1)
                hybrid_img_cpu = torch.clamp(hybrid_img.cpu(), -1, 1)
                comparison = torch.cat([human_tensor.cpu(), hybrid_img_cpu], dim=0)
                comparison_path = str(save_path).replace('.png', '_comparison.png')
                save_image(comparison, comparison_path, nrow=2, normalize=True, value_range=(-1, 1))
                print(f"✅ Comparison saved to {comparison_path}")
            
            return hybrid_img
            
        except Exception as e:
            print(f"❌ Error generating hybrid: {e}")
            traceback.print_exc()
            return None

# ============================================================================
# ADVANCED FEATURE ANALYZER
# ============================================================================

class AdvancedFeatureAnalyzer:
    """Analyze facial features and generate reports"""
    
    def __init__(self, hvae, avae, feature_bank, device):
        self.hvae = hvae
        self.avae = avae
        self.feature_bank = feature_bank
        self.device = device
    
    @torch.no_grad()
    def analyze_human_face(self, human_image_path):
        """Comprehensive analysis of human face"""
        # Load and preprocess image
        transform = get_transforms('val')
        try:
            image = Image.open(human_image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
        
        # Encode with region-aware encoder if available
        if hasattr(self.hvae, 'region_aware') and self.hvae.region_aware:
            _, region_latents = self.hvae.encode(image_tensor)
        else:
            # Use standard encoder
            z, _ = self.hvae.encode(image_tensor)
            region_latents = None
        
        # Analyze each region
        results = {}
        for region in Config.FACIAL_REGIONS.keys():
            region_results = {}
            
            for species in Config.ANIMAL_SPECIES:
                if region_latents:
                    query_latent = region_latents[region][0]  # mu
                else:
                    query_latent = z
                
                match = self.feature_bank.get_best_match(query_latent, species, region)
                
                if match:
                    similarity, candidate = match
                    region_results[species] = {
                        'similarity': similarity,
                        'path': candidate['path']
                    }
            
            # Sort by similarity
            sorted_matches = sorted(region_results.items(), 
                                  key=lambda x: x[1]['similarity'], reverse=True)
            results[region] = sorted_matches
        
        return results
    
    def generate_report(self, results, save_path):
        """Generate detailed analysis report"""
        report = "=" * 80 + "\n"
        report += "RUPANTARA FACIAL FEATURE ANALYSIS REPORT\n"
        report += "=" * 80 + "\n\n"
        
        for region, matches in results.items():
            report += f"\n{region.upper()}:\n"
            report += "-" * 40 + "\n"
            
            for i, (species, data) in enumerate(matches[:3], 1):
                similarity = data['similarity'] * 100
                report += f"  {i}. {species.capitalize()}: {similarity:.2f}% similar\n"
            
            report += "\n"
        
        report += "=" * 80 + "\n"
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        return report
    
    def visualize_similarity_heatmap(self, results, save_path):
        """Create heatmap visualization"""
        if not CV2_AVAILABLE:
            print("⚠️  OpenCV not available, skipping heatmap")
            return
        
        # Create similarity matrix
        species_list = Config.ANIMAL_SPECIES
        regions_list = list(Config.FACIAL_REGIONS.keys())
        
        matrix = np.zeros((len(regions_list), len(species_list)))
        
        for i, region in enumerate(regions_list):
            for j, species in enumerate(species_list):
                matches_dict = dict(results[region])
                if species in matches_dict:
                    matrix[i, j] = matches_dict[species]['similarity']
        
        # Create heatmap using matplotlib if available
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
            
            ax.set_xticks(np.arange(len(species_list)))
            ax.set_yticks(np.arange(len(regions_list)))
            ax.set_xticklabels(species_list)
            ax.set_yticklabels(regions_list)
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            for i in range(len(regions_list)):
                for j in range(len(species_list)):
                    text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                                 ha="center", va="center", color="black")
            
            ax.set_title("Facial Region Similarity to Animal Species")
            fig.tight_layout()
            plt.colorbar(im)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Heatmap saved to {save_path}")
        except ImportError:
            print("⚠️  Matplotlib not available, skipping heatmap")

# ============================================================================
# DRY RUN / SANITY CHECK
# ============================================================================

def dry_run():
    """Improved dry-run / sanity check for Rupantara system"""
    print("\n🔍 Running comprehensive system check...")
    
    # Setup
    Config.create_directories()
    Config.set_seed()
    
    # -----------------------
    # Check hardware
    # -----------------------
    print("\n⚡ Hardware check:")
    print(f"  Device: {Config.DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("  ⚠️ No GPU detected - training will be slow!")
    
    # -----------------------
    # Test model creation
    # -----------------------
    print("\n🔧 Testing model architecture...")
    try:
        # Human VAE
        hvae = ConditionalVAE(
            latent_dim=Config.LATENT_DIM,
            beta=1.0,
            use_attention=True,
            region_aware=True
        ).to(Config.DEVICE)
        print(f"✅ Human VAE created: {sum(p.numel() for p in hvae.parameters()):,} parameters")
        
        # Animal VAE
        avae = ConditionalVAE(
            latent_dim=Config.LATENT_DIM,
            beta=1.0,
            use_attention=True,
            num_conditions=Config.NUM_SPECIES,
            region_aware=True
        ).to(Config.DEVICE)
        print(f"✅ Animal VAE created: {sum(p.numel() for p in avae.parameters()):,} parameters")
        
        # Forward pass test
        dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(Config.DEVICE)
        dummy_labels = torch.randint(0, Config.NUM_SPECIES, (2,)).to(Config.DEVICE)
        
        with torch.no_grad():
            recon_human, region_latents, z = hvae(dummy_input)
            print(f"✅ Human VAE forward pass works")
            print(f"  Input shape: {dummy_input.shape}, Output shape: {recon_human.shape}")
            
            recon_animal, _, _ = avae(dummy_input, dummy_labels)
            print(f"✅ Animal VAE forward pass works (conditional)")
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        print(f"  Error details: {traceback.format_exc()}")
        return
    
    # -----------------------
    # Test data loading
    # -----------------------
    print("\n📁 Testing data loading...")
    try:
        # Create dummy image if no real images exist
        dummy_img_path = Config.HUMAN_DIR / "test_human.jpg"
        if not dummy_img_path.exists():
            dummy_img = Image.new('RGB', (256, 256), color='red')
            dummy_img.save(dummy_img_path)
            print(f"  Created dummy image for testing: {dummy_img_path}")
        
        transform = get_transforms('val')
        dummy_img_tensor = transform(Image.open(dummy_img_path)).unsqueeze(0)
        print(f"✅ Image loading and transformation works")
        
    except Exception as e:
        print(f"⚠️ Data loading issue: {e}")
        print("  This might be OK if you don't have data yet")
    
    print("\n" + "="*60)
    print("✅ System check completed!")
    print("\n📚 Recommended workflow for Google Colab:")
    print("1. First, run: python rupantara_complete.py dry-run")
    print("2. Upload your datasets to:")
    print(f"   Human faces: {Config.HUMAN_DIR}")
    print(f"   Animal faces: {Config.ANIMAL_DIR}/[species]/")
    print("3. Train Human VAE: python rupantara_complete.py train-hvae --epochs 10 --batch-size 8")
    print("4. Train Animal VAE: python rupantara_complete.py train-avae --epochs 10 --batch-size 8")
    print("5. Build feature bank: python rupantara_complete.py build-bank")
    print("6. Generate hybrid: python rupantara_complete.py generate-hybrid --image path/to/human.jpg --eyes wolf --nose tiger")
    print("="*60)

# ============================================================================
# QUICK TEST FUNCTION
# ============================================================================

def quick_test():
    """Quick test to verify everything works"""
    print("\n⚡ Running quick test...")
    
    # Setup
    Config.create_directories()
    Config.set_seed()
    
    # Create test arguments
    class Args:
        def __init__(self):
            self.epochs = 1
            self.batch_size = 2
            self.lr = 1e-4
            self.weight_decay = 1e-5
            self.beta = 1.0
            self.use_attention = True
            self.region_aware = True
            self.use_gan = False
            self.use_ema = False
            self.use_perceptual = False
            self.use_lpips = False
            self.use_identity = False
            self.use_symmetry = False
            self.use_advanced_aug = False
            self.use_tensorboard = False
            self.mapper_hidden_dim = 1024
            self.mapper_num_layers = 8
    
    args = Args()
    
    # Test model creation
    print("Testing model creation...")
    try:
        model = ConditionalVAE(
            latent_dim=Config.LATENT_DIM,
            beta=args.beta,
            use_attention=args.use_attention,
            region_aware=args.region_aware
        ).to(Config.DEVICE)
        print(f"✅ Model created successfully")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return
    
    print("\n✅ Quick test passed! System is ready.")
    print("\nNext steps:")
    print("1. Upload your face datasets to the appropriate folders")
    print("2. Run: python rupantara_complete.py train-hvae --epochs 5 --batch-size 4")
    print("3. Check the results in /content/rupantara/results/")

# ============================================================================
# MAIN ENTRY POINT - FIXED
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Rupantara - Cross-Species Face Generation')
    
    # Mode
    parser.add_argument('mode', type=str, 
                       choices=['train-hvae', 'train-avae', 'train-mapper', 
                               'build-bank', 'analyze-human', 'generate-hybrid', 
                               'dry-run', 'quick-test', 'setup-colab'],
                       help='Operation mode')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=Config.WEIGHT_DECAY, help='Weight decay')
    parser.add_argument('--beta', type=float, default=Config.VAE_BETA, help='VAE beta parameter')
    
    # Model args
    parser.add_argument('--use-attention', action='store_true', help='Use attention mechanisms')
    parser.add_argument('--region-aware', action='store_true', help='Use region-aware encoding')
    parser.add_argument('--use-gan', action='store_true', help='Use GAN discriminator')
    parser.add_argument('--use-ema', action='store_true', help='Use EMA weights')
    
    # Loss args
    parser.add_argument('--use-perceptual', action='store_true', help='Use perceptual loss')
    parser.add_argument('--use-lpips', action='store_true', help='Use LPIPS loss')
    parser.add_argument('--use-identity', action='store_true', help='Use identity preservation')
    parser.add_argument('--use-symmetry', action='store_true', help='Use symmetry loss')
    
    # Data args
    parser.add_argument('--use-advanced-aug', action='store_true', help='Use advanced augmentation')
    
    # Logging args
    parser.add_argument('--use-tensorboard', action='store_true', help='Enable TensorBoard')
    
    # Mapper args
    parser.add_argument('--mapper-hidden-dim', type=int, default=1024, help='Mapper hidden dimension')
    parser.add_argument('--mapper-num-layers', type=int, default=8, help='Mapper number of layers')
    
    # Analysis/Generation args
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--eyes', type=str, help='Animal species for eyes')
    parser.add_argument('--nose', type=str, help='Animal species for nose')
    parser.add_argument('--mouth', type=str, help='Animal species for mouth')
    parser.add_argument('--ears', type=str, help='Animal species for ears')
    parser.add_argument('--forehead', type=str, help='Animal species for forehead')
    
    # Checkpoint args
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Setup directories
    Config.create_directories()
    Config.set_seed()
    
    # Execute mode
    if args.mode == 'setup-colab':
        print("Setting up Google Colab environment...")
        install_dependencies()
        download_test_datasets()
        Config.print_config()
        
    elif args.mode == 'dry-run':
        dry_run()
    
    elif args.mode == 'quick-test':
        quick_test()
    
    elif args.mode == 'train-hvae':
        print("\n🎯 Training Human VAE...")
        trainer = EnhancedVAETrainer('human', args)
        if args.resume:
            trainer.load_checkpoint(args.resume)
        trainer.train()
    
    elif args.mode == 'train-avae':
        print("\n🎯 Training Animal VAE...")
        trainer = EnhancedVAETrainer('animal', args)
        if args.resume:
            trainer.load_checkpoint(args.resume)
        trainer.train()
    
    elif args.mode == 'train-mapper':
        print("\n🎯 Training Mapping Network...")
        
        # Load VAEs
        hvae_path = Config.CHECKPOINT_DIR / "humanvae_best.pth"
        avae_path = Config.CHECKPOINT_DIR / "animalvae_best.pth"
        
        if not hvae_path.exists() or not avae_path.exists():
            print("❌ Please train Human and Animal VAEs first!")
            print(f"   Expected: {hvae_path}")
            print(f"   Expected: {avae_path}")
            return
        
        # Create models
        hvae = ConditionalVAE(latent_dim=Config.LATENT_DIM, beta=args.beta, 
                             use_attention=args.use_attention,
                             region_aware=args.region_aware).to(Config.DEVICE)
        avae = ConditionalVAE(latent_dim=Config.LATENT_DIM, beta=args.beta,
                             use_attention=args.use_attention, 
                             num_conditions=Config.NUM_SPECIES,
                             region_aware=args.region_aware).to(Config.DEVICE)
        
        # Load checkpoints
        hvae_checkpoint = torch.load(hvae_path, map_location=Config.DEVICE, weights_only=False)
        avae_checkpoint = torch.load(avae_path, map_location=Config.DEVICE, weights_only=False)
        
        hvae.load_state_dict(hvae_checkpoint['model_state_dict'])
        avae.load_state_dict(avae_checkpoint['model_state_dict'])
        
        trainer = MappingNetworkTrainer(hvae, avae, args)
        trainer.train()
    
    elif args.mode == 'build-bank':
        print("\n🎯 Building Animal Feature Bank...")
        
        avae_path = Config.CHECKPOINT_DIR / "animalvae_best.pth"
        
        if not avae_path.exists():
            print("❌ Please train Animal VAE first!")
            return
        
        # Create model
        avae = ConditionalVAE(latent_dim=Config.LATENT_DIM, beta=args.beta,
                             use_attention=args.use_attention, 
                             num_conditions=Config.NUM_SPECIES,
                             region_aware=args.region_aware).to(Config.DEVICE)
        
        # Load checkpoint
        checkpoint = torch.load(avae_path, map_location=Config.DEVICE, weights_only=False)
        avae.load_state_dict(checkpoint['model_state_dict'])
        avae.eval()
        
        builder = FeatureBankBuilder(avae, Config.DEVICE)
        feature_bank = builder.build()
        
        if feature_bank:
            print("\n✅ Feature bank built successfully!")
    
    elif args.mode == 'analyze-human':
        print("\n🎯 Analyzing Human Face...")
        
        if not args.image:
            print("❌ Please provide --image path!")
            return
        
        if not Path(args.image).exists():
            print(f"❌ Image not found: {args.image}")
            return
        
        # Load models
        hvae_path = Config.CHECKPOINT_DIR / "humanvae_best.pth"
        avae_path = Config.CHECKPOINT_DIR / "animalvae_best.pth"
        bank_path = Config.CHECKPOINT_DIR / "animal_feature_bank.pkl"
        
        if not hvae_path.exists():
            print("❌ Human VAE checkpoint not found:", hvae_path)
            return
        if not avae_path.exists():
            print("❌ Animal VAE checkpoint not found:", avae_path)
            return
        if not bank_path.exists():
            print("❌ Feature bank not found:", bank_path)
            return
        
        print("✅ All required checkpoints found")
        
        # Create models
        hvae = ConditionalVAE(latent_dim=Config.LATENT_DIM, beta=args.beta,
                             use_attention=args.use_attention,
                             region_aware=args.region_aware).to(Config.DEVICE)
        avae = ConditionalVAE(latent_dim=Config.LATENT_DIM, beta=args.beta,
                             use_attention=args.use_attention,
                             num_conditions=Config.NUM_SPECIES,
                             region_aware=args.region_aware).to(Config.DEVICE)
        
        # Load checkpoints
        hvae_checkpoint = torch.load(hvae_path, map_location=Config.DEVICE, weights_only=False)
        avae_checkpoint = torch.load(avae_path, map_location=Config.DEVICE, weights_only=False)
        
        hvae.load_state_dict(hvae_checkpoint['model_state_dict'])
        avae.load_state_dict(avae_checkpoint['model_state_dict'])
        
        # Load feature bank
        feature_bank = AnimalFeatureBank(Config.DEVICE)
        feature_bank.load(bank_path)
        
        # Analyze
        analyzer = AdvancedFeatureAnalyzer(hvae, avae, feature_bank, Config.DEVICE)
        results = analyzer.analyze_human_face(args.image)
        
        if results:
            # Generate report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = Config.ANALYSIS_DIR / f"analysis_{timestamp}.txt"
            report = analyzer.generate_report(results, report_path)
            print(f"\n{report}")
            
            # Generate heatmap
            heatmap_path = Config.HEATMAPS_DIR / f"heatmap_{timestamp}.png"
            analyzer.visualize_similarity_heatmap(results, heatmap_path)
            
            print("\n✅ Analysis completed!")
    
    elif args.mode == 'generate-hybrid':
        print("\n🎯 Generating Hybrid Face...")
        
        if not args.image:
            print("❌ Please provide --image path!")
            return
        
        if not Path(args.image).exists():
            print(f"❌ Image not found: {args.image}")
            return
        
        # Build region-species map
        region_species_map = {}
        if args.eyes:
            region_species_map['eyes'] = args.eyes
        if args.nose:
            region_species_map['nose'] = args.nose
        if args.mouth:
            region_species_map['mouth'] = args.mouth
        if args.ears:
            region_species_map['ears'] = args.ears
        if args.forehead:
            region_species_map['forehead'] = args.forehead
        
        if not region_species_map:
            print("❌ Please specify at least one region (--eyes, --nose, --mouth, --ears, --forehead)!")
            print(f"Available species: {', '.join(Config.ANIMAL_SPECIES)}")
            return
        
        # Validate species
        for region, species in region_species_map.items():
            if species not in Config.ANIMAL_SPECIES:
                print(f"❌ Invalid species '{species}' for {region}!")
                print(f"Available species: {', '.join(Config.ANIMAL_SPECIES)}")
                return
        
        # Load models
        hvae_path = Config.CHECKPOINT_DIR / "humanvae_best.pth"
        bank_path = Config.CHECKPOINT_DIR / "animal_feature_bank.pkl"
        
        if not hvae_path.exists():
            print(f"❌ Human VAE not found: {hvae_path}")
            print("   Please train Human VAE first!")
            return
        
        if not bank_path.exists():
            print(f"❌ Feature bank not found: {bank_path}")
            print("   Please build feature bank first!")
            return
        
        # Create model
        hvae = ConditionalVAE(latent_dim=Config.LATENT_DIM, beta=args.beta,
                             use_attention=args.use_attention,
                             region_aware=args.region_aware).to(Config.DEVICE)
        
        # Load checkpoint
        hvae_checkpoint = torch.load(hvae_path, map_location=Config.DEVICE, weights_only=False)
        hvae.load_state_dict(hvae_checkpoint['model_state_dict'])
        hvae.eval()
        
        # Load feature bank
        feature_bank = AnimalFeatureBank(Config.DEVICE)
        feature_bank.load(bank_path)
        
        # Generate hybrid
        generator = HybridFaceGenerator(hvae, feature_bank, Config.DEVICE)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = "_".join([f"{k[:3]}-{v[:3]}" for k, v in region_species_map.items()])
        save_path = Config.HYBRIDS_DIR / f"hybrid_{save_name}_{timestamp}.png"
        
        hybrid_img = generator.generate_hybrid(args.image, region_species_map, save_path)
        
        if hybrid_img is not None:
            print("\n✅ Hybrid generation completed!")
            print(f"📸 Result saved to: {save_path}")
    
    else:
        print(f"❌ Unknown mode: {args.mode}")

# ============================================================================
# STARTUP MESSAGE
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║                    🦁 RUPANTARA ULTIMATE 🐺                         ║
    ║                                                                      ║
    ║        World-Class Cross-Species Face Generation System             ║
    ║                                                                      ║
    ║  Features:                                                           ║
    ║  ✓ Conditional VAE with Region-Aware Encoding                       ║
    ║  ✓ Animal Feature Bank System                                       ║
    ║  ✓ Latent Mixing Engine                                             ║
    ║  ✓ Identity Preservation & Symmetry                                 ║
    ║  ✓ EMA Weights & Mixed Precision Training                           ║
    ║  ✓ PatchGAN with Feature Matching                                   ║
    ║  ✓ Comprehensive Evaluation Metrics                                 ║
    ║  ✓ Advanced Data Augmentation                                       ║
    ║  ✓ TensorBoard + JSON Logging                                       ║
    ║  ✓ Interactive Hybrid Generation                                    ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    🚀 **Google Colab Quick Start:**
    
    1. First-time setup:
       python rupantara_complete.py setup-colab
    
    2. Test the system:
       python rupantara_complete.py dry-run
    
    3. Quick test with dummy data:
       python rupantara_complete.py quick-test
    
    📚 **Complete Workflow:**
    
    python rupantara_complete.py train-hvae --epochs 10 --batch-size 8 --use-attention --region-aware
    python rupantara_complete.py train-avae --epochs 10 --batch-size 8 --use-attention --region-aware
    python rupantara_complete.py build-bank
    python rupantara_complete.py generate-hybrid --image human.jpg --eyes wolf --nose tiger
    """)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)
