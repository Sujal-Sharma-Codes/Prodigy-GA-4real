# Prodigy-GA-4real
# 🏙️ Task-04: Image-to-Image Translation with Pix2Pix (cGAN)

An implementation of **Pix2Pix**, a conditional generative adversarial network (cGAN) that translates one type of image into another — in this case, building facade **label maps** into realistic **facade photos**.

Completed as part of the **Generative AI Internship at Prodigy InfoTech**.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-GPU-red) ![Colab](https://img.shields.io/badge/Google-Colab-orange)

---

## 🎯 Objective
Implement an image-to-image translation model using a conditional GAN (Pix2Pix), which learns a mapping from an input image to a corresponding output image using paired training data.

## 🧠 How It Works
Pix2Pix is a **conditional GAN**: unlike a standard GAN that generates images from random noise, the generator is conditioned on an input image and learns to produce the matching output image.

Two networks are trained together:
- **Generator (U-Net):** an encoder-decoder with skip connections between mirrored layers, so fine spatial detail from the input reaches the output directly instead of being lost by compressing through a bottleneck.
- **Discriminator (PatchGAN):** instead of judging the whole image as real or fake in one shot, it classifies overlapping patches of the image. This pushes the generator to get local texture and detail right, not just the overall layout.

**Loss function:** the generator is trained with two combined losses:
- **Adversarial loss** — fool the discriminator into thinking the output is real
- **L1 (pixel-wise) loss**, weighted 100x heavier — keep the output close to the actual target image, not just "realistic-looking"

## 🛠️ Implementation
| Component | Choice |
|---|---|
| Dataset | CMP Facades (400 train / 100 val paired images), from the original Pix2Pix paper |
| Generator | U-Net, 8 down-sampling + 7 up-sampling blocks with skip connections |
| Discriminator | PatchGAN (70x70 receptive field) |
| Image size | 256 x 256 |
| Adversarial loss | MSE (LSGAN-style, more stable than plain BCE) |
| L1 weight (λ) | 100 |
| Optimizer | Adam, lr 2e-4, betas (0.5, 0.999) |
| Epochs | 20 |
| Hardware | Google Colab, NVIDIA T4 GPU |

## 🖼️ Results
Each sample grid shows, left to right: **input label map → generated photo → real photo**.

| Epoch | Result |
|---|---|
| 5 | ![epoch 5](outputs/epoch_5.png) |
| 10 | ![epoch 10](outputs/epoch_10.png) |
| 15 | ![epoch 15](outputs/epoch_15.png) |
| 20 | ![epoch 20](outputs/epoch_20.png) |

As training progresses, the generated photos go from blurry, washed-out color blocks to recognisable facade textures — windows, doors and wall shading roughly aligned with the input label map.

## 💡 Key Learnings
- **Paired data changes everything.** Because Pix2Pix has an exact input-output pair for every training example, it can use a direct L1 loss, which trains much faster and more stably than an unconditional GAN.
- **PatchGAN focuses on texture, not layout.** Since it only judges local patches, the overall structure comes from the L1 loss and skip connections, while PatchGAN sharpens fine detail.
- **The L1/adversarial balance matters.** A high L1 weight (100) keeps output close to the ground truth; too low and the model prioritizes fooling the discriminator over pixel accuracy.
- **GAN training needs care.** I hit a tensor-shape mismatch between the discriminator's actual output size and a hand-calculated patch size — a reminder to infer shapes from the model at runtime rather than computing them by formula.

## ⚠️ Limitations
- 20 epochs on ~400 image pairs gives visibly correct structure but still-soft textures; the original paper trains for 200 epochs for sharper results.
- Pix2Pix requires **paired** data — the input and target must correspond exactly. For translation between unpaired image sets (e.g. horses ↔ zebras), CycleGAN is the standard alternative.
- Results are specific to the facades domain; a new domain (e.g. sketches → photos) needs its own paired dataset and retraining.

## ▶️ How to Run
1. Open `pix2pix_image_translation_v2.ipynb` in [Google Colab](https://colab.research.google.com).
2. Set **Runtime → Change runtime type → T4 GPU**.
3. Run all cells in order. Training takes roughly 15-25 minutes for 20 epochs.
4. Sample image grids are saved to `outputs/` every 5 epochs.

To train on your own paired dataset, replace the facades download with your own image pairs, keeping each file as a single image with input and target side by side (or adjust the `Dataset` class to load them from separate folders).

## 📁 Project Structure
```
├── pix2pix_image_translation_v2.ipynb
├── outputs/
│   ├── epoch_5.png
│   ├── epoch_10.png
│   ├── epoch_15.png
│   └── epoch_20.png
├── pix2pix_generator.pth
└── README.md
```

## 📚 References
- Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks* (Pix2Pix paper, 2017)
- [CMP Facades dataset](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/)
- [Official Pix2Pix project page](https://phillipi.github.io/pix2pix/)
- 
