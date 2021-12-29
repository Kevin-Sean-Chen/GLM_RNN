%Opto_landscape
function M = Opto_landscape(Xdim, Ydim, Xc, Yc, Width, Amplitude, Background, Noise, Note, invert)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xdim dimension of the x-axis (columns in image)
% Ydim dimension of the y-axis (rows in image)
% Xc location of the center of Gaussian in x-axis
% Yc location of the center of Gaussian in y-axis
% Landscape parameters: (1/Z)*((x-Xc).^2 + (y-Yc).^2)./(2*Width^2) * Amplitude + Background + Noise
% Note is a string added to the image name
% invert is a binary logic (0,1) for whether or not the landscape is inverted (1 for inverting to have a Gaussian upside-down)

%%% Example:
%for a 900x570 image, center at (450,285), with width=100, max=255, background 20, and noise level=2, save with note "test"
% M = Opto_landscape(900, 570, 450, 285, 100, 255, 20, 2, 'test', 0);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%parameters
center = [Yc, Xc];  %center location
D = Width;  %width parameter
Cs = Amplitude;  %max value from odor source
Cb = Background;  %constant background
M0 = zeros(Ydim, Xdim);  %initial matrix for correct image size
Sn = Noise;  %Gaussian white noise amplitude
M = Gauss2D(M0, D, center, Cs, Cb, Sn);  %generate landscape

%IF inverse landscape
if invert==1
    M = abs(M - max(M(:)));
end

%rectify for image in unit8, then normalize to unit ratio for RGB image
%M(M>255) = 255;
M = uint8(M);
bmap = zeros(256,3);
bmap(:,3) = [0:255]/255;
M_RGB = ind2rgb(M,bmap);

%plot and save as image
figure; imagesc(M_RGB)
imwrite(M_RGB, ['odor_landscape_', Note ,'.tif'])

end

%Gaussian values
function val = Gaussian(x, y, sigma, center)
xc = center(1);
yc = center(2);
exponent = ((x-xc).^2 + (y-yc).^2)./(2*sigma^2);
amplitude = 1/(sigma*sqrt(2*pi));
val = amplitude * exp(-exponent);  %Gaussian form
end

%build 2D Gaussian landscape
function mat = Gauss2D(mat0, sigma, center, Cs, Cb, Sn)
gsize = size(mat0);
[R,C] = ndgrid(1:gsize(1), 1:gsize(2));
mat_unit = Gaussian(R, C, sigma, center);
mat_unit = mat_unit/max(mat_unit(:));  %rescale according to the max value
effective_Cs = Cs-Cb;
mat = mat_unit*effective_Cs + Cb + Sn*randn(gsize(1), gsize(2));  %rescale by the max-amplitude and add background and noise
end
