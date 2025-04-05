img1 = imread('EnhanceAfter.jpg'); 
figure;
imshow(img1)
title("original img");
img = imresize(img1, [256, 256]); 
disp(['Data type of the image: ', class(img)])
figure;
imshow(img)
title("Resized img")
%% 
% Title: Histogram of RGB Channels
figure
subplot(3,1,1);
r = img(:,:,1)
imhist(r); title('Red Channel');
subplot(3,1,2);
g = img(:,:,2)
imhist(g); title('Green Channel');
subplot(3,1,3);
b = img(:,:,3)
imhist(b); title('Blue Channel');
%% 
% Title: Grayscale Images
% (Ir + Ig + Ib)/3

gray1 = (r+g+b)/3; 
% (Ir/3 + Ig/3 + Ib/3)
gray2 = (r/3 + g/3 + b/3); 

% The RGB values are converted to grayscale using the NTSC formula: 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue.
gray3 = 0.299*(r) + 0.587*(g) + 0.114*(b)
figure; 
subplot(1,3,1); 
imshow(uint8(gray1)); 
title('(Ir + Ig + Ib)/3');
subplot(1,3,2); 
imshow(uint8(gray2)); 
title('Sum (Ir/3 + Ig/3 + Ib/3)');
subplot(1,3,3); 
imshow(uint8(gray3)); 
title('0.299*(r) + 0.587*(g) + 0.114*(b)');
%% 
% Title: Grayscale Histogram grayscale 
figure
subplot(3,1,1);
histogram(gray1); title('grayscale1');
disp(['Min grayscale 1 level: ', num2str(min(gray1(:)))]);
disp(['Max grayscale 1 level: ', num2str(max(gray1(:)))]);
disp(['Data type of grayscale 1 image: ', class(gray1)]);

subplot(3,1,2);
histogram(gray2); title('grayscale2');
disp(['Min grayscale 2 level: ', num2str(min(gray2(:)))]);
disp(['Max grayscale 2 level: ', num2str(max(gray2(:)))]);
disp(['Data type of grayscale 2 image: ', class(gray2)]);

subplot(3,1,3);
histogram(gray3); title('grayscale3');
disp(['Min grayscale 3 level: ', num2str(min(gray3(:)))]);
disp(['Max grayscale 3 level: ', num2str(max(gray3(:)))]);
disp(['Data type of grayscale 3 image: ', class(gray3)]);


%% 
% Generate Gaussian white noise
noise_mean = 0; noise_std = 20; % Parameters
noise = noise_mean + noise_std * randn(size(gray2)); 
figure

imshow(noise);
title('Gaussian White Noise');

%% 

figure 

histogram(noise(:),50);
title('Histogram of Noise');

noise_mean_calc = mean(noise(:));

noise_std_calc = std(noise(:));
disp(['Noise mean: ', num2str(noise_mean_calc)]);
disp(['Noise std: ', num2str(noise_std_calc)]);



%% 

noisy_img = double(gray2) + noise;
figure

imshow(noisy_img,[]);
title('Noisy Image');

figure 
histogram(noisy_img);
title('Histogram of Noisy Image');
disp(['Data type of noisy image: ', class(noisy_img)]);






%%
% Explanation of why the autocorrelation image appears as a single white dot:
%
% 1. **Autocorrelation Properties**:
%    - The autocorrelation of white noise is ideally a **delta function**,
%      meaning it has a strong peak at the center (zero shift) and near-zero values elsewhere.
%    - This results in a **high-intensity central peak** (white dot) with near-zero values
%      elsewhere (black background).
%
% 2. **Behavior of xcorr2(noisy_img)**:
%    - `xcorr2` computes the **full 2D cross-correlation** of the input matrix.
%    - The result is an output of size: (2M-1, 2N-1), where MxN is the original image size.
%    - This means the output matrix is larger than the original image, and most values are 
%      close to zero except for the peak in the middle.
%
% 3. **Why Only a White Dot?**
%    - The central peak is much higher than all other values.
%    - When using `imshow(noise_autocorr, [])`, MATLAB normalizes the display contrast.
%    - As a result, the small values appear black, and only the high-intensity peak remains visible.
%
% SOLUTIONS:
% - Normalize the autocorrelation result before visualization.
% - Use a 3D surface plot to observe the correlation structure.
% - Plot a 1D cross-section through the center of the autocorrelation matrix.

% -------------------------------------------------------------
% **Mathematical Working of xcorr2**
%
% The 2D cross-correlation (which xcorr2 computes) is defined as:
%
%               C(m, n) = ∑∑ A(i, j) * B(i - m, j - n)
%
% where:
% - A(i, j) is the first matrix (image or signal).
% - B(i, j) is the second matrix (usually the same image for autocorrelation).
% - The summation runs over all valid indices.
%
% In the case of autocorrelation:
% - `xcorr2(A)` computes the correlation of `A` with itself.
% - The peak at the center represents the highest correlation (where the image aligns with itself).
% - Away from the center, the values decrease, indicating less correlation.


%% 
% Autocorrelation of noise
noise_autocorr = xcorr2(noise); % Autocorrelation
figure 
imshow(noise_autocorr, []);
figure
imhist(noise_autocorr)

% title('Autocorrelation of Noise image');
% figure 
% 
% imhist(noise_autocorr);
% title('Histogram of noise_autocorr');


%% 

% Alternative visualization
figure;
surf(noise_autocorr, 'EdgeColor', 'none'); % 3D surface plot
title('3D View of Noise Autocorrelation');
colormap jet; shading interp;

%% 

% Apply average filtering with two different kernel sizes
h1 = fspecial('average', [5 5]);% 3x3 Gaussian filter
h2 = fspecial('average', [15 15]); % 7x7 Gaussian filter

filtered_img1 = conv2(noisy_img, h1, 'same');
filtered_img2 = conv2(noisy_img, h2, 'same');

% Compute SNR
noise_img_std = std(noise(:));

% SNR for 3x3 filter
signal_max1 = max(double(filtered_img1(:)));
signal_min1 = min(double(filtered_img1(:)));
SNR1 = (signal_max1 - signal_min1) / noise_img_std;
PSNR1 = 20 * log10(SNR1);

% SNR for 7x7 filter
signal_max2 = max(double(filtered_img2(:)));
signal_min2 = min(double(filtered_img2(:)));
SNR2 = (signal_max2 - signal_min2) / noise_img_std;
PSNR2 = 20 * log10(SNR2);

% Display results
figure;
subplot(1,3,1); imshow(uint8(noisy_img)); title('Noisy Image');
subplot(1,3,2); imshow(uint8(filtered_img1)); title(['Filtered 3x3 SNR: ', num2str(SNR1)]);
subplot(1,3,3); imshow(uint8(filtered_img2)); title(['Filtered 15x15 SNR: ', num2str(SNR2)]);

% Display SNR values
disp(['SNR (3x3 filter): ', num2str(SNR1)]);
disp(['PSNR (3x3 filter) in dB: ', num2str(PSNR1)]);
disp(['SNR (7x7 filter): ', num2str(SNR2)]);
disp(['PSNR (7x7 filter) in dB: ', num2str(PSNR2)]);



%% 

% Compute SNR and PSNR
noise_img_std= std(noise(:)); 
h = fspecial('average', [5 5]); %  smoothing filter (5x5 average)
filtered_noisy = conv2(noisy_img, h, 'same');
filtered_gray2 = conv2(gray2, h, 'same');

 %filtered_signal = conv2(gray2, h, 'same');
signal_max1 = max(double(filtered_noisy(:)));
signal_min1 = min(double(filtered_noisy(:)));
SNR = (signal_max1 - signal_min1) / noise_img_std;
PSNR_dB = 20 * log10(SNR);
disp(['SNR of filtered noisy image: ', num2str(SNR)]);
disp(['PSNR (dB) of filtered noisy image: ', num2str(PSNR_dB)]);

 %filtered_signal = conv2(gray2, h, 'same');
signal_max2 = max(double(filtered_gray2(:)));
signal_min2 = min(double(filtered_gray2(:)));
SNR = (signal_max2 - signal_min2) / noise_img_std;
PSNR_dB = 20 * log10(SNR);
disp(['SNR of  grayscale image ', num2str(SNR)]);
disp(['PSNR (dB) of grayscale image: ', num2str(PSNR_dB)]);

%% 

% Define filter sizes (odd values: 3x3, 5x5, ...,)
filter_sizes = 3:2:50; % Odd sizes
snr_values = zeros(size(filter_sizes)); % Store SNR values
psnr_values = zeros(size(filter_sizes)); % Store PSNR values

for i = 1:length(filter_sizes)
    % Apply smoothing filter
    h = fspecial('average', [filter_sizes(i), filter_sizes(i)]); % Create filter
    filtered_img = conv2(noisy_img, h, 'same'); % Apply filter
    
    % Compute SNR using your formula
    signal_max = max(double(filtered_img(:)));
    signal_min = min(double(filtered_img(:)));
    SNR = (signal_max - signal_min) / noise_std;
    snr_values(i) = SNR;
    snr_values(i)
    % Compute PSNR
    PSNR_dB = 20 * log10(SNR);
    psnr_values(i) = PSNR_dB;
end

% Plot SNR vs. Filter Size
figure;
plot(filter_sizes, snr_values, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Filter Size (NxN)');
ylabel('SNR');
title('Effect of Filter Size on SNR');
grid on;

% Plot PSNR vs. Filter Size
figure;
plot(filter_sizes, psnr_values, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'Color', 'r');
xlabel('Filter Size (NxN)');
ylabel('PSNR (dB)');
title('Effect of Filter Size on PSNR');
grid on;

%% 
% Step 12: Nonlinear noise reduction performance
%  using median filtering
nonlinear_filtered = medfilt2(noisy_img, [3 3]); % Median filter
figure(11); % Title: Nonlinear Filtered Image
imshow(nonlinear_filtered, []);
title('Nonlinear Filtered Image');





%% Apply median filter and recompute SNR


noise_img_std= std(noisy_img(:)); 
signal_max = max(double(gray2(:)));
signal_min = min(double(gray2(:)));
SNR = (signal_max - signal_min) / noise_img_std;
PSNR_dB = 20 * log10(SNR);

disp(['SNR: ', num2str(SNR)]);
disp(['PSNR (dB): ', num2str(PSNR_dB)]);

%% 


filtered_noisy = medfilt2(noisy_img, [10 10]); % Apply 3x3 median filter
% filtered_signal = medfilt2(gray2, [3 3]);   % Apply median filter on original signal
filter_noise_std = std(filtered_noisy(:));
% Compute new SNR after applying median filter
new_SNR = (max(double(filtered_noisy(:))) - min(double(filtered_noisy(:)))) / filter_noise_std;
disp(['New SNR after median filtering: ', num2str(new_SNR)]);
new_PSNR_dB = 20 * log10(double(new_SNR));
disp(['new PSNR (dB): ', num2str(PSNR_dB)]);

%% 


% Apply Nonlinear Median Filtering with different kernel sizes
filtered_median_3x3 = medfilt2(noisy_img, [3 3]); % 3x3 filter
filtered_median_7x7 = medfilt2(noisy_img, [7 7]); % 7x7 filter

% Compute SNR
noise_img_std = std(noisy_img(:));

% SNR for 3x3 Median Filter
signal_max1 = max(double(filtered_median_3x3(:)));
signal_min1 = min(double(filtered_median_3x3(:)));
SNR1 = (signal_max1 - signal_min1) / noise_img_std;
PSNR1 = 20 * log10(SNR1);

% SNR for 7x7 Median Filter
signal_max2 = max(double(filtered_median_7x7(:)));
signal_min2 = min(double(filtered_median_7x7(:)));
SNR2 = (signal_max2 - signal_min2) / noise_img_std;
PSNR2 = 20 * log10(SNR2);

% Display Results
figure;
subplot(1,3,1); imshow(uint8(noisy_img)); title('Noisy Image');
subplot(1,3,2); imshow(uint8(filtered_median_3x3)); title(['Median 3x3 SNR: ', num2str(SNR1)]);
subplot(1,3,3); imshow(uint8(filtered_median_7x7)); title(['Median 7x7 SNR: ', num2str(SNR2)]);

% Display SNR values
disp(['SNR (3x3 Median Filter): ', num2str(SNR1)]);
disp(['PSNR (3x3 Median Filter) in dB: ', num2str(PSNR1)]);
disp(['SNR (7x7 Median Filter): ', num2str(SNR2)]);
disp(['PSNR (7x7 Median Filter) in dB: ', num2str(PSNR2)]);


