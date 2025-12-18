# # 🦁 Rupantara : Unsupervised Cross-Species Face Generation and Reverse Feature Mapping

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**RUPANTARA** is an advanced deep learning system for generating hybrid human-animal facial features using Variational Autoencoders (VAEs), region-aware encoding, and intelligent latent space mixing.

---

## 🌟 Features

- ✅ **Conditional VAE Architecture** - Separate models for human and animal faces
- ✅ **Region-Aware Encoding** - Processes 5 facial regions independently (eyes, nose, mouth, ears, forehead)
- ✅ **Animal Feature Bank** - Stores and retrieves 12,970+ facial features from 8 species
- ✅ **Intelligent Latent Mixing** - Combines human and animal features seamlessly
- ✅ **Self-Attention Mechanisms** - Captures long-range dependencies in facial features
- ✅ **Multiple Loss Functions** - MSE, KL divergence, perceptual, identity preservation, symmetry
- ✅ **Comprehensive Metrics** - FID, SSIM, PSNR, LPIPS evaluation
- ✅ **Similarity Heatmaps** - Visual analysis of facial region similarities
- ✅ **Google Colab Optimized** - Runs efficiently on free Colab instances

---

## 🐾 Supported Animal Species

- 🐺 **Wolf** - Piercing eyes and sharp features
- 🐯 **Tiger** - Distinctive stripes and powerful presence
- 🦁 **Lion** - Majestic and commanding features
- 🐆 **Leopard** - Spotted patterns and sleek design
- 🐻 **Bear** - Strong, robust facial structure
- 🦊 **Fox** - Sharp, intelligent features
- 🐕 **Dog** - Friendly, expressive characteristics
- 🐱 **Cat** - Graceful and refined features

---

## 📊 Model Architecture

### Human VAE
- **Parameters**: 23,105,093
- **Latent Dimension**: 512
- **Region Latent Dimension**: 128 per region
- **Architecture**: 5-layer encoder/decoder with self-attention
- **Training**: 10 epochs, validation loss: 0.129

### Animal VAE
- **Parameters**: 25,202,757
- **Latent Dimension**: 512
- **Conditional**: Species-specific embeddings
- **Training**: 10 epochs, validation loss: 0.134

### Feature Bank
- **Total Features**: 12,970
- **Structure**: 8 species × 5 regions × ~300 samples
- **Storage**: Cosine similarity-based retrieval

---

## 🚀 Quick Start

### 1. Installation (Google Colab)

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install -q lpips albumentations opencv-python scikit-image

# Clone/upload your code
!cp /path/to/rupantara_complete.py /content/drive/MyDrive/rupantara/
```

### 2. Dataset Setup

Organize your dataset as follows:

```
/content/drive/MyDrive/rupantara/data/
├── humans/
│   ├── human_001.jpg
│   ├── human_002.jpg
│   └── ... (3000+ images)
└── animals/
    ├── cat/
    │   ├── cat_001.jpg
    │   └── ... (400 images)
    ├── dog/
    │   ├── dog_001.jpg
    │   └── ... (400 images)
    ├── wolf/
    │   ├── wolf_001.jpg
    │   └── ... (300 images)
    ├── tiger/
    │   ├── tiger_001.jpg
    │   └── ... (299 images)
    ├── lion/
    ├── leopard/
    ├── bear/
    └── fox/
```

### 3. Training Pipeline

```bash
# Step 1: Test system
python rupantara_complete.py dry-run

# Step 2: Train Human VAE
python rupantara_complete.py train-hvae     --epochs 10     --batch-size 8     --use-attention     --region-aware

# Step 3: Train Animal VAE
python rupantara_complete.py train-avae     --epochs 10     --batch-size 8     --use-attention     --region-aware

# Step 4: Build Feature Bank
python rupantara_complete.py build-bank     --use-attention     --region-aware
```

### 4. Generate Hybrids

```bash
# Generate hybrid face
python rupantara_complete.py generate-hybrid     --image "/path/to/human.jpg"     --eyes wolf     --nose tiger     --mouth lion     --use-attention     --region-aware
```

### 5. Analyze Faces

```bash
# Generate similarity heatmap
python rupantara_complete.py analyze-human     --image "/path/to/human.jpg"     --use-attention     --region-aware
```

---

## 📈 Training Results

### Human VAE Performance

| Epoch | Train Loss | Val Loss | SSIM | PSNR |
|-------|-----------|----------|------|------|
| 1 | 0.2298 | 0.1754 | 0.72 | 18.4 |
| 5 | 0.1509 | 0.1426 | 0.81 | 21.2 |
| 9 | 0.1332 | 0.1291 | 0.85 | 23.1 |
| 10 | 0.1326 | 0.1319 | 0.84 | 22.8 |

### Animal VAE Performance

| Epoch | Train Loss | Val Loss | Species Balance |
|-------|-----------|----------|-----------------|
| 1 | 0.2150 | 0.1777 | Balanced |
| 5 | 0.1571 | 0.1470 | Balanced |
| 10 | 0.1389 | 0.1342 | Balanced |

---

## 🎨 Usage Examples

### Example 1: Full Face Transformation

```python
!python rupantara_complete.py generate-hybrid     --image "celebrity.jpg"     --eyes wolf     --nose tiger     --mouth lion     --ears bear     --forehead fox     --use-attention     --region-aware
```

**Output**: Hybrid face with wolf eyes, tiger nose, lion mouth, bear ears, and fox forehead

**Similarity Scores**:
- Wolf Eyes: 32.4%
- Tiger Nose: 32.0%
- Lion Mouth: 20.6%

---

### Example 2: Subtle Canine Features

```python
!python rupantara_complete.py generate-hybrid     --image "portrait.jpg"     --eyes wolf     --nose fox     --use-attention     --region-aware
```

**Output**: Subtle wolf eyes and fox nose, keeping other features human

---

### Example 3: Big Cat Transformation

```python
!python rupantara_complete.py generate-hybrid     --image "face.jpg"     --eyes tiger     --nose leopard     --mouth lion     --use-attention     --region-aware
```

**Output**: Fierce predator features with tiger eyes, leopard nose, lion mouth

---

## 📁 Project Structure

```
rupantara/
├── rupantara_complete.py       # Main system code
├── data/                        # Training datasets
│   ├── humans/
│   └── animals/
├── logs/                        # Training logs & checkpoints
│   ├── checkpoints/
│   │   ├── humanvae_best.pth
│   │   ├── animalvae_best.pth
│   │   └── animal_feature_bank.pkl
│   └── tensorboard/
├── results/                     # Generated outputs
│   ├── hybrids/                # Hybrid faces
│   ├── reconstructions/        # VAE reconstructions
│   └── analysis/               # Heatmaps & reports
│       └── heatmaps/
└── README.md
```

---

## 🔧 Advanced Configuration

### Training Parameters

```python
# Customize training
python rupantara_complete.py train-hvae     --epochs 50     --batch-size 16     --lr 1e-4     --beta 1.0     --use-attention     --region-aware     --use-gan \              # Enable GAN discriminator
    --use-ema \              # Exponential moving average
    --use-perceptual \       # VGG perceptual loss
    --use-identity \         # Face recognition loss
    --use-symmetry \         # Facial symmetry loss
    --use-tensorboard        # Enable TensorBoard logging
```

### Model Architectures

```python
# Standard VAE (faster, less detailed)
python rupantara_complete.py train-hvae     --epochs 10     --batch-size 8

# Advanced VAE (slower, more detailed)
python rupantara_complete.py train-hvae     --epochs 30     --batch-size 4     --use-attention     --region-aware     --use-gan     --use-perceptual
```

---

## 📊 Evaluation Metrics

The system evaluates generated faces using:

- **FID (Fréchet Inception Distance)**: Measures distribution similarity
- **SSIM (Structural Similarity)**: Compares structural patterns
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction quality
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Perceptual quality
- **Cosine Similarity**: Feature matching accuracy

---

## 🎯 Best Practices

### 1. Data Quality
- Use high-quality, front-facing images (256×256 recommended)
- Ensure good lighting and clear facial features
- Balance dataset across species (300-400 images per species)

### 2. Training Tips
- Start with 10 epochs for quick testing
- Use `--use-attention --region-aware` for best quality
- Monitor validation loss to avoid overfitting
- Save checkpoints every 2-5 epochs

### 3. Hybrid Generation
- Use 2-3 animal features for subtle results
- Use 4-5 features for dramatic transformations
- Experiment with different species combinations
- Check similarity scores (>25% recommended)

### 4. Troubleshooting
- If hybrid is blurry: Train for more epochs
- If hybrid is blank: Check latent value ranges
- If training is slow: Reduce batch size or disable advanced features
- If out of memory: Use smaller batch size (4-8)

---

## 🐛 Common Issues & Solutions

### Issue 1: Blank/Blue Hybrid Images
**Solution**: Ensure proper value clamping in decoder output
```python
hybrid_img = torch.clamp(hybrid_img, -1, 1)
```

### Issue 2: PyTorch 2.6 Loading Error
**Solution**: Add `weights_only=False` to `torch.load()`
```python
checkpoint = torch.load(path, map_location=device, weights_only=False)
```

### Issue 3: Low GPU Memory
**Solution**: Reduce batch size and disable heavy features
```bash
python rupantara_complete.py train-hvae     --batch-size 4     --epochs 10
```

### Issue 4: Poor Hybrid Quality
**Solution**: Train longer with advanced losses
```bash
python rupantara_complete.py train-hvae     --epochs 30     --use-attention     --region-aware     --use-perceptual     --use-identity
```

---

## 📝 Research & Theory

### Variational Autoencoder (VAE)
- Learns compressed latent representation of faces
- Enables smooth interpolation in latent space
- KL divergence ensures latent space structure

### Region-Aware Encoding
- Processes 5 facial regions independently
- Captures region-specific features (eyes, nose, etc.)
- Enables fine-grained control over hybrid generation

### Feature Bank System
- Stores animal facial features in latent space
- Uses cosine similarity for feature matching
- Retrieves best-matching animal features for each region

### Latent Mixing
- Combines human and animal latent codes
- Preserves facial structure while adding animal features
- Weighted mixing based on region importance

---

## 🎓 Academic Context

This system implements concepts from:

- **VAE Theory**: Kingma & Welling (2013) - "Auto-Encoding Variational Bayes"
- **Perceptual Loss**: Johnson et al. (2016) - "Perceptual Losses for Real-Time Style Transfer"
- **Self-Attention**: Zhang et al. (2019) - "Self-Attention Generative Adversarial Networks"
- **Feature Matching**: Salimans et al. (2016) - "Improved Techniques for Training GANs"

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **More Animal Species**: Add birds, reptiles, marine animals
2. **3D Face Models**: Extend to 3D facial geometry
3. **Video Support**: Generate animated hybrid faces
4. **Style Transfer**: Add artistic style options
5. **Mobile App**: Deploy as mobile application
6. **Web Interface**: Create browser-based UI

---

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset Sources**: 
  - Human faces: CelebA, FFHQ datasets
  - Animal faces: ImageNet, AFHQ datasets
- **Frameworks**: PyTorch, torchvision
- **Tools**: Google Colab, TensorBoard
- **Libraries**: PIL, NumPy, scikit-image, matplotlib, seaborn

---

## 📧 Contact

For questions, issues, or collaborations:

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/rupantara/issues)
- **Email**: your.email@example.com
- **Twitter**: @yourhandle

---

## 🌟 Citation

If you use RUPANTARA in your research, please cite:

```bibtex
@software{rupantara2024,
  title={RUPANTARA: Cross-Species Face Generation System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/rupantara}
}
```

---

## 📈 Roadmap

### Version 1.0 (Current)
- ✅ Human & Animal VAE training
- ✅ Region-aware encoding
- ✅ Feature bank system
- ✅ Hybrid generation
- ✅ Similarity analysis

### Version 2.0 (Planned)
- ⏳ Improved latent mixing with trained networks
- ⏳ Real-time generation
- ⏳ Web interface
- ⏳ More animal species (20+)
- ⏳ Video support

### Version 3.0 (Future)
- ⏳ 3D face models
- ⏳ Mobile application
- ⏳ API endpoints
- ⏳ Cloud deployment
- ⏳ Commercial licensing

---

**Made with ❤️ by the RUPANTARA Team**

*Transform faces, blend species, create the extraordinary.* 🦁🐺🐯

---

Last Updated: December 14, 2024