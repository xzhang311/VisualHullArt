# VisualHullArt

## Overview
VisualHullArt is a Python project that generates artistic 3D reconstructions using the visual hull technique. It processes silhouette images from multiple viewpoints to create 3D voxel models, which are then rendered with artistic effects for unique visual outputs. This repository is ideal for enthusiasts of computer vision, 3D modeling, and digital art.

## Features
- **Visual Hull Reconstruction**: Constructs 3D voxel models from 2D silhouette images.
- **Artistic Rendering**: Applies effects like watercolor, sketch, or neon to 3D models.
- **Multi-View Processing**: Supports input from multiple camera angles for accurate reconstructions.
- **Export Options**: Saves outputs as 3D models (OBJ) or rendered images (PNG/JPG).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/xzhang311/VisualHullArt.git
   ```
2. Navigate to the project directory:
   ```bash
   cd VisualHullArt
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main script to generate a 3D model and apply an artistic effect:
```bash
python main.py --silhouettes path/to/silhouette_folder --output path/to/output.obj --effect sketch
```
Available effects include `sketch`, `watercolor`, `neon`, and more. Use the `--help` flag for details:
```bash
python main.py --help
```

## Requirements
- Python 3.8+
- Libraries: NumPy, OpenCV, VTK, Matplotlib (listed in `requirements.txt`)

## Contributing
Contributions are welcome! Fork the repository, create a feature branch, and submit a pull request with your enhancements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
Built with inspiration from research in visual hull algorithms and open-source 3D rendering tools.
