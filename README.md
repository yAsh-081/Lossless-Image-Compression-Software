# 🖼️ Lossless Image Compression Software

A powerful, multi-stage lossless image compression tool that combines advanced algorithms to achieve exceptional compression ratios without any quality loss. Built with Python and featuring an intuitive GUI, this tool makes professional-grade image compression accessible to everyone.

![Application Screenshot](images/app_screenshot.png)
*Main application interface*

## ✨ Features

- **🎯 Multi-Stage Compression Pipeline**: Combines Paeth prediction, Run-Length Encoding (RLE), and Huffman coding for optimal results
- **📊 Impressive Compression Ratios**: Achieves up to 77% compression across various BMP image types (4-24 bit/pixel)
- **🎨 Wide Format Support**: Handles multiple BMP formats including 4-bit, 8-bit, and 24-bit per pixel
- **📈 Real-Time Performance Metrics**: Live tracking of file sizes, compression ratios, and processing time
- **🚀 Zero Crashes**: Comprehensive edge case handling ensures stable operation
- **💾 Efficient Storage**: Optimized Huffman tree serialization reduces overhead by 40-80%
- **🧠 Smart Algorithm Selection**: Automatically chooses the best compression techniques for each image type
- **🖥️ User-Friendly GUI**: Clean Tkinter interface with intuitive controls

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yAsh-081/Lossless-Image-Compression-Software.git
cd Lossless-Image-Compression-Software
```

2. Ensure you have Python 3.7+ installed (no additional dependencies required - uses only Python standard library)

### Usage

1. Launch the application:
```bash
python main.py
```

2. **Select an image**: Click the "Browse" or "Select Image" button to choose a BMP image

3. **Compress**: Click the "Compress" button to start the compression process

4. **View Results**: The application will display:
   - Original file size
   - Compressed file size
   - Compression ratio percentage
   - Processing time

5. **Save Compressed File**: Save your compressed image with the `.compressed` extension

6. **Decompress** (optional): Load a compressed file and decompress it to restore the original image

![Compression Process](images/compression_demo.png)
*Compression process demonstration*

## 🔧 How It Works

The compression pipeline uses a sophisticated three-stage approach:

### 1. Paeth Prediction
Predicts pixel values based on neighboring pixels (left, top, and top-left), reducing data redundancy by storing only the prediction errors.

### 2. Run-Length Encoding (RLE)
Efficiently encodes sequences of identical values, particularly effective for images with uniform regions or patterns.

### 3. Huffman Coding
Assigns variable-length codes to symbols based on their frequency, with more common values receiving shorter codes for optimal compression.

### Dynamic Technique Selection
The software intelligently analyzes each image and selects the optimal combination of techniques, improving average compression ratios by 15-25% across diverse image types.

## 📊 Performance

![Test Results](images/test_results.png)
*Compression performance across different image types*

- **Average Compression Ratio**: 40-77% depending on image content
- **Processing Speed**: Real-time compression for images up to 10MB
- **Memory Efficiency**: Optimized algorithms minimize RAM usage
- **Tree Storage Overhead**: Reduced by 60-80% through compact binary format

## 🛠️ Technical Details

### Supported Formats
- BMP (4-bit per pixel)
- BMP (8-bit per pixel)
- BMP (24-bit per pixel)

### Compression Format
Compressed files use a custom binary format (`.compressed`) that includes:
- Compressed image data
- Huffman tree structure (optimized serialization)
- Metadata for perfect reconstruction

### Edge Cases Handled
- Empty or corrupted image files
- Images with unusual dimensions
- Edge pixels without complete neighbor sets
- Memory constraints for large files
- Invalid file formats

## 📁 Project Structure

```
Lossless-Image-Compression-Software/
├── main.py    # Complete application (GUI + algorithms)
├── test/                  # Test images (8 sample BMP files)
│   ├── image1.bmp
│   ├── image2.bmp
│   └── ...
└── README.md
```

## 🧪 Testing

The `test/` folder contains 8 sample BMP images for testing the compression algorithm on different image types and complexities. Simply load any of these images through the GUI to see the compression results.

![Test Results](images/test_results.png)
*Compression performance across the 8 test images*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Paeth prediction algorithm from the PNG specification
- Huffman coding theory by David A. Huffman
- Inspiration from modern lossless compression standards

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Author**: Yash  
**Repository**: [github.com/yAsh-081/Lossless-Image-Compression-Software](https://github.com/yAsh-081/Lossless-Image-Compression-Software)