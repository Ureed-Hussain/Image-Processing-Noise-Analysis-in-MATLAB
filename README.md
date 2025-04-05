________________________________________
📊 Image Processing & Noise Analysis in MATLAB
This project provides a comprehensive walkthrough of essential image processing tasks using MATLAB. It covers color space analysis, grayscale conversion, noise generation, autocorrelation visualization, and performance evaluation of noise reduction filters using Signal-to-Noise Ratio (SNR) and Peak Signal-to-Noise Ratio (PSNR).
________________________________________
📁 Contents
1.	📷 Image Loading and Preprocessing
2.	🌈 RGB Histogram Visualization
3.	⚫ Grayscale Conversion Methods
4.	📉 Grayscale Histogram Analysis
5.	🎲 Gaussian White Noise Generation
6.	🌀 Autocorrelation of Noisy Image
7.	🧹 Noise Reduction Using Filters
8.	📈 SNR & PSNR Analysis
9.	🧮 Filter Performance vs. Size
10.	⚙️ Nonlinear Median Filtering
________________________________________
🔧 Requirements
•	MATLAB (R2021a or newer recommended)
•	Image Processing Toolbox
________________________________________
🧪 Features & Explanation
1. Image Preprocessing
•	Loads the input image.
•	Resizes to 256×256 pixels for uniform analysis.
2. Color Channel Histograms
•	Extracts R, G, and B channels.
•	Displays histograms for color intensity distribution.
3. Grayscale Conversion Techniques
•	Arithmetic mean: (R + G + B) / 3
•	Averaged RGB components: (R/3 + G/3 + B/3)
•	NTSC standard: 0.299*R + 0.587*G + 0.114*B
4. Grayscale Histograms
•	Displays histograms for each grayscale variant.
•	Prints min, max, and data types.
5. Gaussian Noise Generation
•	Adds Gaussian white noise to grayscale image.
•	Mean and standard deviation of the noise are computed and visualized.
6. Autocorrelation Analysis
•	Explains and visualizes the autocorrelation of noise.
•	Includes a 3D surface plot for structure inspection.
7. Filtering Techniques
•	Applies average filters (5×5, 15×15) to noisy images.
•	Measures improvement using:
o	SNR: Signal-to-Noise Ratio
o	PSNR: Peak Signal-to-Noise Ratio (in dB)
8. Filter Size Impact
•	Analyzes how filter size affects noise reduction.
•	Plots SNR and PSNR against filter size (3x3 to 49x49).
9. Nonlinear Filtering
•	Uses a median filter to remove impulse noise.
•	Recomputes and compares SNR and PSNR post-filtering.
________________________________________
📊 Sample Outputs
•	RGB and grayscale histograms
•	Noisy vs. filtered images
•	Autocorrelation visualization
•	SNR & PSNR plots
•	Median filter results
________________________________________
📚 Learnings
This project demonstrates:
•	RGB to grayscale transformation techniques
•	Effects of Gaussian noise on images
•	Power of linear and nonlinear filters
•	Quantitative evaluation using SNR & PSNR
•	Insightful use of xcorr2 and histogram-based validation
________________________________________
🙌 Acknowledgements
Developed as part of coursework in Image Processing. Inspired by classic filtering and statistical noise analysis techniques.

