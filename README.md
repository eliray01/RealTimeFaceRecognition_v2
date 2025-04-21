
# RealTimeFaceRecognition_v2

RealTimeFaceRecognition_v2 is a Python-based application designed for real-time face recognition using deep learning techniques.
It integrates object detection and instance segmentation to identify and label faces in live video streams or images.

## Features

- Real-time face detection and recognition
- Instance segmentation for precise object boundaries
- Utilizes Mask R-CNN architecture for object detection
- Modular code structure for easy maintenance and scalability

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.6 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/eliray01/RealTimeFaceRecognition_v2.git
   cd RealTimeFaceRecognition_v2
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

   *Note: If `requirements.txt` is not present, manually install the necessary packages as indicated in the code files.*

## Usage

1. **Run the main application:**

   ```bash
   python main.py
   ```

   This will start the real-time face recognition application using your default webcam.

2. **Training the model:**

   To train the model with custom data:

   ```bash
   python train_v2.py
   ```

   Ensure your training data is properly formatted and placed in the appropriate directories.

3. **Instance segmentation on images:**

   To perform instance segmentation on a set of images:

   ```bash
   python mask_rcnn_images.py
   ```

   Modify the script as needed to specify the input and output directories.

## Project Structure

```
RealTimeFaceRecognition_v2/
├── architecture.py
├── coco_names.py
├── encodings/
├── instance_segmentation.py
├── main.py
├── mask_rcnn_images.py
├── train_v2.py
├── utils.py
├── __pycache__/
├── .idea/
└── requirements.txt
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/YourFeature`
5. Open a pull request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Mask R-CNN](https://github.com/matterport/Mask_RCNN)
- COCO Dataset
