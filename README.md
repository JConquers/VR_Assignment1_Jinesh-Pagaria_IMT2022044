# Visual Recognition Assignment: Coin Detection and Image Stitching

## Overview
This project involves two main computer vision tasks:

### Part 1: Coin Detection, Segmentation, and Counting


*Input:* An image containing scattered Indian coins.

### Part 2: Image Stitching to Create a Panorama
*Input:* 3 images with overlapping regions.

## Folder Structure
```
project-root
├── input               # Folder for input images
├── output1a            # Output images for image transformations
├── output1b            # Output images for coin edge detection
├── output1c            # Output images for separated
├── output2             # Output images for stitched panorama
└── scripts             # Python scripts for processing
```

## Setup and Execution
1. **Clone the Repository:**
```
git clone <repository-url>
cd <project-folder>
```
2. **Set Up Virtual Environment:**
```
python3 -m venv env
source env/bin/activate  # For Linux/Mac
env\Scripts\activate     # For Windows
```
3. **Install Required Libraries:**
```
pip install -r requirements.txt
```
4. **Run Coin Detection and Segmentation:**
```
python scripts/coin_detection.py
```
5. **Run Image Stitching:**
```
python scripts/image_stitching.py
```
6. **View Outputs:**
- Coin image transformation `output1a`
- Coin edge detection in `output1b`
- Separated coins images in `output1c`
- Stitched panorama in `output2`

## Requirements
- Python 3.8+
- OpenCV
- NumPy
- Matplotlib



