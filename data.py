from re import I
import av
import cv2
from joblib import Parallel, delayed
import numpy as np
import os
import pylnk3
import random
import torch
import torch.functional as F
import torch.nn as nn

from augmennt import augmennt as transforms
from augmennt.augmennt import functional as TF
from color_modem.image import ImageModem
from color_modem.line import LineConfig
from color_modem.color.niir import NiirModem
from color_modem.color.pal import PalDModem, PalSModem
from color_modem.color.secam import SecamModem
from PIL import Image, ImageFilter

#random.seed(0)
#torch.manual_seed(0)

VIDEO_CODECS = ["mpeg2video", "mpeg4"]

class VariableRandomCrop(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes
    
    def forward(self, img):
        h, w = self.sizes[torch.randint(0, len(self.sizes), (1,)).item()]
        if img.shape[0] < h:
            h = img.shape[0] - (img.shape[0] % 8)
        if img.shape[1] < w:
            w = img.shape[1] - (img.shape[1] % 8)

        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=(h, w))

        return TF.crop(img, i, j, h, w)

class RandomResize(nn.Module):
    def __init__(self, min_scale=0.5, max_scale=2.0, interpolation=None):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.interpolation = interpolation
    
    def forward(self, img):
        scale = random.uniform(self.min_scale, self.max_scale)
        h = int(img.shape[0] * scale)
        w = int(img.shape[1] * scale)

        interpolation = self.interpolation
        if not interpolation:
            # cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR, cv2.INTER_NEAREST
            interpolation = torch.randint(0, 5, (1,)).item()
        
        return cv2.resize(img, (w, h), interpolation=interpolation)

class RandomDownscale(nn.Module):
    def __init__(self, min_height=128, max_height=256, interpolation=None):
        super().__init__()
        self.min_height = min_height
        self.max_height = max_height
        self.interpolation = interpolation
    
    def forward(self, img):
        if img.shape[0] <= self.min_height:
            return img
        
        h = torch.randint(self.min_height, self.max_height, (1,)).item()
        h = min(h, img.shape[0])
        aspect_ratio = img.shape[1] / img.shape[0]
        w = int(h * aspect_ratio)

        interpolation = self.interpolation
        if not interpolation:
            # cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR, cv2.INTER_NEAREST
            interpolation = torch.randint(0, 5, (1,)).item()
        
        return cv2.resize(img, (w, h), interpolation=interpolation)
    

class RelativeCenterCrop(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes
    
    def forward(self, img):
        h = img.shape[0] - (self.sizes[0] * 2)
        w = img.shape[1] - (self.sizes[1] * 2)

        return TF.center_crop(img, (h, w))
    
class RoundedCrop(nn.Module):
    def __init__(self, round_to=8):
        super().__init__()
        self.round_to = round_to
    
    def forward(self, img):
        h = img.shape[0] - (img.shape[0] % self.round_to)
        w = img.shape[1] - (img.shape[1] % self.round_to)

        return TF.center_crop(img, (h, w))

class NtscColor(nn.Module):
    def __init__(self, p, modems):
        super().__init__()
        self.p = p
        self.modems = modems
    
    def forward(self, img):
        if self.p < random.random():
            return img

        old_shape = img.shape
        # resize to a random height between 432 and 628
        height = torch.randint(432, 628, (1,)).item()
        img = cv2.resize(img, (int(img.shape[1] * (height / img.shape[0])), height))

        img = Image.fromarray(img)
        i = 0
        while True:
            try:
                line_config = LineConfig(img.size)
                modem = self.modems[torch.randint(0, len(self.modems), (1,)).item()](line_config)
            except:
                print(f"failed to create modem {i} . . .")
                i += 1
                continue
            break
        img_modem = ImageModem(modem)
        
        frame = torch.randint(0, 20, (1,)).item()
        img = img_modem.demodulate(img_modem.modulate(img, frame), frame)

        # resize back to original size
        img = np.asarray(img)
        img = cv2.resize(img, (old_shape[1], old_shape[0]))

        return img
        
class Rescale(nn.Module):
    def __init__(self, scale, interpolation=None):
        super().__init__()
        self.scale = scale
        self.interpolation = interpolation
    
    def forward(self, img):
        size=(int(img.shape[1] * self.scale), int(img.shape[0] * self.scale))
        interpolation = self.interpolation
        if not interpolation:
            # cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR, cv2.INTER_NEAREST
            interpolation = torch.randint(0, 5, (1,)).item()

        img = cv2.resize(img, size, interpolation=interpolation)
        return img

class To16Bit(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        if img.dtype == np.uint16:
            return img
        return (img*256).astype('uint16')

class To8Bit(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        if img.dtype == np.uint8:
            return img
        return (img/256).astype('uint8')

class RandomVariableRescale(nn.Module):
    def __init__(self, p = 0.5, sizes=[(720, 540)], interpolation1=None, interpolation2=None):
        super().__init__()
        self.p = p
        self.sizes = sizes
        self.interpolation1 = interpolation1
        self.interpolation2 = interpolation2
        
    def forward(self, img):
        if self.p < random.random():
            return img

        original_size = (img.shape[1], img.shape[0])
        scaled_size = self.sizes[torch.randint(0, len(self.sizes), (1,)).item()]

        interpolation1 = self.interpolation1
        if not interpolation1:
            # cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR, cv2.INTER_NEAREST
            interpolation1 = torch.randint(0, 5, (1,)).item()
        interpolation2 = self.interpolation2
        if not interpolation2:
            interpolation2 = torch.randint(0, 5, (1,)).item()

        img = cv2.resize(img, scaled_size, interpolation=interpolation1)
        img = cv2.resize(img, original_size, interpolation=interpolation2)
        return img

class RandomRescaleBlur(nn.Module):
    def __init__(self, p=0.5, scale=[0.5,2.0], interpolation1=None, interpolation2=None):
        super().__init__()
        self.p = p
        self.scale = scale
        self.interpolation1 = interpolation1
        self.interpolation2 = interpolation2

    def forward(self, img):
        if self.p < random.random():
            return img

        original_size = (img.shape[1], img.shape[0])
        scale = float(torch.empty(1).uniform_(self.scale[0], self.scale[1]))

        size=(int(img.shape[1] * scale), int(img.shape[0] * scale))

        interpolation1 = self.interpolation1
        if not interpolation1:
            # cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR, cv2.INTER_NEAREST
            interpolation1 = torch.randint(0, 5, (1,)).item()
        interpolation2 = self.interpolation2
        if not interpolation2:
            interpolation2 = torch.randint(0, 5, (1,)).item()

        img = cv2.resize(img, size, interpolation=interpolation1)
        img = cv2.resize(img, original_size, interpolation=interpolation2)
        return img

class RandomDeinterlace(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        if self.p < random.random():
            return img

        img = cv2.resize(img, (img.shape[1], img.shape[0]//2), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (img.shape[1], img.shape[0] *2), interpolation=cv2.INTER_LINEAR)

        return img

class RandomUnsharp(nn.Module):
    def __init__(self, p=0.5, strength=[0.3, 0.8]):
        super().__init__()
        self.p = p
        self.unsharp = transforms.FilterUnsharp('median',
            strength=float(torch.empty(1).uniform_(strength[0], strength[1])))

    def forward(self, img):
        if self.p < random.random():
            return img

        return self.unsharp(img)

class RandomInvert(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        if self.p < random.random():
            return img

        return cv2.bitwise_not(img)

class RandomChannelShuffle(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        if self.p < random.random():
            return img
        
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

class RandomCrush(nn.Module):
    def __init__(self, p=0.5, in_black=0, in_white=255):
        super().__init__()
        self.p = p
        self.in_black = in_black
        self.in_white = in_white

    def forward(self, img):
        if self.p < random.random():
            return img
        
        inBlack  = np.array([self.in_black, self.in_black, self.in_black], dtype=np.float32)
        inWhite  = np.array([self.in_white, self.in_white, self.in_white], dtype=np.float32)
        inGamma  = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        outBlack = np.array([0, 0, 0], dtype=np.float32)
        outWhite = np.array([255, 255, 255], dtype=np.float32)

        img = np.clip( (img - inBlack) / (inWhite - inBlack), 0, 255 )                            
        img = ( img ** (1/inGamma) ) *  (outWhite - outBlack) + outBlack
        img = np.clip( img, 0, 255).astype(np.uint8)

        return img

class RandomGhosting(nn.Module):
    def __init__(self, p=0.5, range=2, opacity=[0.1, 0.5]):
        super().__init__()
        self.p = p
        self.range = range
        self.opacity = opacity

    def forward(self, img):
        if self.p < random.random():
            return img

        opacity = float(torch.empty(1).uniform_(self.opacity[0], self.opacity[1]))
        x = torch.randint(-self.range, self.range+1, (1,)).item()
        y = torch.randint(-self.range, self.range+1, (1,)).item()

        base = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        overlay = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        overlay[...,3] = 255 * opacity

        translation_matrix = np.float32([ [1,0,x], [0,1,y] ])
        overlay = cv2.warpAffine(overlay, translation_matrix, (img.shape[1], img.shape[0]))
        img = cv2.addWeighted(base, 0.5, overlay, 0.5, 0.0)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        return img

def rand_roll_channel(channel, range=1):
    x = torch.randint(-range, range+1, (1,)).item()
    y = torch.randint(-range, range+1, (1,)).item()

    axes = [None, 0, 1, (0, 1)]
    axis = axes[torch.randint(0, len(axes), (1,)).item()]

    return np.roll(channel, (x, y), axis=axis)

class RandomRGBColorShift(nn.Module):
    def __init__(self, p=0.5, range=0.01):
        super().__init__()
        self.p = p
        self.range = range

    def forward(self, img):
        if self.p < random.random():
            return img
        
        max_value = (2 ** (img.itemsize * 8) - 1)
        r, g, b = cv2.split(img.astype(np.float32) / max_value)

        r += float(torch.empty(1).uniform_(-self.range, self.range))
        b += float(torch.empty(1).uniform_(-self.range, self.range))
        g += float(torch.empty(1).uniform_(-self.range, self.range))
        #noise = (np.ones_like(img) - (np.random.rand(img.shape[0], img.shape[1], img.shape[2]) * 2)) * self.range

        img = (cv2.merge((r, g, b)) * max_value).round(0).clip(0, max_value).astype(img.dtype)
        return img    

class RandomYUVColorShift(nn.Module):
    def __init__(self, p=0.5, range=0.01):
        super().__init__()
        self.p = p
        self.range = range

    def forward(self, img):
        if self.p < random.random():
            return img
        
        max_value = (2 ** (img.itemsize * 8) - 1)
        y, u, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2YUV).astype(np.float32) / max_value)

        y += float(torch.empty(1).uniform_(-self.range, self.range))
        u += float(torch.empty(1).uniform_(-self.range, self.range))
        v += float(torch.empty(1).uniform_(-self.range, self.range))
        #noise = (np.ones_like(img) - (np.random.rand(img.shape[0], img.shape[1], img.shape[2]) * 2)) * self.range * 0.1

        img = (cv2.merge((y, u, v)) * max_value).round(0).clip(0, max_value).astype(img.dtype)
        return cv2.cvtColor(img, cv2.COLOR_YUV2RGB)

class RandomRGBAberration(nn.Module):
    def __init__(self, p=0.5, range=1):
        super().__init__()
        self.p = p
        self.range = range

    def forward(self, img):
        if self.p < random.random():
            return img

        r, g, b = cv2.split(img)

        r = rand_roll_channel(r, self.range)
        g = rand_roll_channel(g, self.range)
        b = rand_roll_channel(b, self.range)

        img = cv2.merge((r, g, b))
        return img

class RandomYCrCbAberration(nn.Module):
    def __init__(self, p=0.5, range=1):
        super().__init__()
        self.p = p
        self.range = range

    def forward(self, img):
        if self.p < random.random():
            return img

        img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(img)

        cr = rand_roll_channel(cr, self.range)
        cb = rand_roll_channel(cb, self.range)

        img = cv2.merge((y, cr, cb))
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

        return img

class RandomYCrCbSubsampling(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        if self.p < random.random():
            return img

        img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(img)

        cr = cv2.resize(cr, (y.shape[1]//2, y.shape[0]//2), interpolation=cv2.INTER_LINEAR)
        cr = cv2.resize(cr, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        cb = cv2.resize(cb, (y.shape[1]//2, y.shape[0]//2), interpolation=cv2.INTER_LINEAR)
        cb = cv2.resize(cb, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR)

        img = cv2.merge((y, cr, cb))
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

        return img

class RandomYUVAberration(nn.Module):
    def __init__(self, p=0.5, range=1):
        super().__init__()
        self.p = p
        self.range = range

    def forward(self, img):
        if self.p < random.random():
            return img

        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        y, u, v = cv2.split(img)

        u = rand_roll_channel(u, self.range)
        v = rand_roll_channel(v, self.range)

        img = cv2.merge((y, u, v))
        img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)

        return img

class RandomYUVSubsampling(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        if self.p < random.random():
            return img

        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        y, u, v = cv2.split(img)

        u = cv2.resize(u, (y.shape[1]//2, y.shape[0]//2), interpolation=cv2.INTER_LINEAR)
        u = cv2.resize(u, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        v = cv2.resize(v, (y.shape[1]//2, y.shape[0]//2), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR)

        img = cv2.merge((y, u, v))
        img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)

        return img

class RandomVideoCompression(nn.Module):
    def __init__(self, p=0.5, bitrate=[100, 2000]):
        super().__init__()
        self.p = p
        self.bitrate = bitrate

    def forward(self, img):
        if self.p < random.random():
            return img
        
        bitrate=int(1000 * float(torch.empty(1).uniform_(self.bitrate[0], self.bitrate[1])))
        codec = VIDEO_CODECS[torch.randint(0, len(VIDEO_CODECS), (1,)).item()]
        filename = f"test_{str(img[0][0][0])}_{bitrate}.ts"

        # setup
        container = av.open(filename, mode="w")
        stream = container.add_stream(codec, rate=1,
            options={'b:v': str(bitrate)})
        stream.width = img.shape[1]
        stream.height = img.shape[0]
        stream.pix_fmt = "yuv420p"

        # encoding
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        # decoding
        container = av.open(filename)
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="rgb24")
            break
        container.close()
        os.remove(filename)

        return img

class ToFrames(nn.Module):
    def __init__(self, n_frames=5):
        super().__init__()
        self.n_frames = n_frames
    
    def forward(self, img):
        return [img] * self.n_frames

class RandomJitter(nn.Module):
    def __init__(self, p=0.5, range=3):
        super().__init__()
        self.p = p
        self.range = range
    
    def forward(self, img):
        if self.p < random.random():
            return img
        
        # shift the image by max range pixels
        x = torch.randint(-self.range, self.range, (1,)).item()
        y = torch.randint(-self.range, self.range, (1,)).item()
        M = np.float32([[1, 0, x], [0, 1, y]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        return img

class PadIfNeeded(nn.Module):
    def __init__(self, width, height, border_mode=cv2.BORDER_REFLECT_101):
        super().__init__()
        self.width = width
        self.height = height
        self.border_mode = border_mode
    
    def forward(self, img):
        if img.shape[0] < self.height or img.shape[1] < self.width:
            pad_width = max(self.width - img.shape[1], 0)
            pad_height = max(self.height - img.shape[0], 0)
            pad_l = pad_width // 2
            pad_r = pad_width - pad_l
            pad_t = pad_height // 2
            pad_b = pad_height - pad_t
            img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, self.border_mode)
            
        return img

IMG_EXTENSIONS = [".bmp", ".jpeg", ".jpg", ".jp2", ".png", ".webp", ".pbm", ".pgm", ".ppm", ".pxm", ".pnm", ".tiff", ".tif", ".exr", ".hdr", ".pic"]
def retrieve_imgs(path):
    if isinstance(path, str):
        paths = [path]
    else:
        paths = path

    imgs = []
    for path in paths:
        if os.path.splitext(path)[1].lower() == ".lnk":
            lnk = pylnk3.parse(path)
            path = lnk.path

        if os.path.isdir(path):
            imgs += retrieve_imgs([os.path.join(path, x) for x in os.listdir(path)])
        elif os.path.splitext(path)[1].lower() in IMG_EXTENSIONS:
            imgs.append(path)

    return imgs


class SingleImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, base_transform, hr_transform, lr_transform):
        self.paths = retrieve_imgs(root)
        self.base_transform = base_transform
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform
        
    def __getitem__(self, index):
        img = cv2.imread(self.paths[index], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # rotate vertical images
        if img.shape[0] > img.shape[1]:
            img = np.rot90(img)

        img = self.base_transform(img)

        if type(img) is list:
            return [self.hr_transform(x) for x in img], [self.lr_transform(x) for x in img]
        else:
            return self.hr_transform(img), self.lr_transform(img)

    def __len__(self):
        return len(self.paths)


BLUR_TRANSFORM = transforms.Compose([
    transforms.RandomChoice([
        transforms.RandomAverageBlur(1.0, 3),
        transforms.RandomBoxBlur(1.0, 3),
        transforms.RandomSincBlur(1.0),
        RandomRescaleBlur(1.0, [0.75, 1.25]),
    ]),
])
NOISE_TRANSFORM = transforms.Compose([
    transforms.RandomChoice([
        transforms.RandomCameraNoise(1.0),
        transforms.RandomGaussianNoise(1.0),
        transforms.RandomPoissonNoise(1.0),
        transforms.RandomSpeckleNoise(1.0),
    ]),
])

SUBSAMPLING_TRANSFORM = transforms.Compose([
    transforms.RandomChoice([
        RandomYCrCbSubsampling(1.0),
        RandomYUVSubsampling(1.0),
    ]),
])
ABERRATION_TRANSFORM = transforms.Compose([
    transforms.RandomChoice([
        RandomRGBAberration(1.0),
        RandomYCrCbAberration(1.0),
        RandomYUVAberration(1.0),
    ]),
])

MAIN_TRANSFORM = transforms.Compose([
    transforms.RandomChoice([
        nn.Identity(),
        BLUR_TRANSFORM,
        NOISE_TRANSFORM,
        transforms.Compose([
            BLUR_TRANSFORM,
            NOISE_TRANSFORM,
        ]),
        transforms.Compose([
            NOISE_TRANSFORM,
            BLUR_TRANSFORM,
        ]),
    ]),
])
COLOR_TRANSFORM = transforms.Compose([
    transforms.RandomChoice([
        nn.Identity(),
        ABERRATION_TRANSFORM,
        SUBSAMPLING_TRANSFORM,
        transforms.Compose([
            ABERRATION_TRANSFORM,
            SUBSAMPLING_TRANSFORM,
        ]),
    ]),
])

def process_j(source_dataset, hr_folder, lr_folder, j, suffix="", n = 1):
    for k in range(n):
        img_filename = os.path.splitext(os.path.basename(source_dataset.paths[j]))[0]
        hr, lr = source_dataset.__getitem__(j)

        if not os.path.exists(hr_folder):
            os.mkdir(hr_folder)
        if not os.path.exists(lr_folder):
            os.mkdir(lr_folder)

        if type(hr) is list:
            hr_img_folder = os.path.join(hr_folder, img_filename)
            lr_img_folder = os.path.join(lr_folder, img_filename)
            if not os.path.exists(hr_img_folder):
                os.mkdir(hr_img_folder)
            if not os.path.exists(lr_img_folder):
                os.mkdir(lr_img_folder)
            for i in range(len(hr)):
                hr[i] = cv2.cvtColor(hr[i], cv2.COLOR_RGB2BGR)
                lr[i] = cv2.cvtColor(lr[i], cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(hr_img_folder, f"{img_filename}_{suffix}_{i}_{k}.png"), hr[i])
                cv2.imwrite(os.path.join(lr_img_folder, f"{img_filename}_{suffix}_{i}_{k}.png"), lr[i])
        else:
            hr = cv2.cvtColor(hr, cv2.COLOR_RGB2BGR)
            lr = cv2.cvtColor(lr, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(hr_folder, f"{img_filename}_{k}.png"), hr)
            cv2.imwrite(os.path.join(lr_folder, f"{img_filename}_{k}.png"), lr)


def main():
    base_transform=transforms.Compose([
        To8Bit(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.25),
        transforms.RandomChoice([
            RandomRGBColorShift(1.0, 0.1),
            RandomYUVColorShift(1.0, 0.1),
        ]),
        RandomChannelShuffle(0.25),
        RandomInvert(0.2),
        #BLUR_TRANSFORM,
        #RandomResize(0.2, 0.5, cv2.INTER_LINEAR),
        
        RoundedCrop(64),
        ToFrames(5), # change for the VSR model's frame range
    ])
    hr_transform=transforms.Compose([
        RelativeCenterCrop([16, 16]),
    ])
    lr_transform=transforms.Compose([
        RandomJitter(1.0, 3),
        transforms.ColorJitter(0.005, 0.005, 0.005, 0.005),
        
        #RandomCrush(1.0, 32, 224),
        RandomUnsharp(0.2),
        
        BLUR_TRANSFORM,
        NOISE_TRANSFORM,
        
        Rescale(0.5),

        NtscColor(0.4, [PalSModem, PalDModem, NiirModem]),

        transforms.RandomChoice([
            RandomVideoCompression(1.0),
            transforms.RandomCompression(1.0, 20, 90),
            nn.Identity(),
        ]),

        RandomDeinterlace(0.25),

        COLOR_TRANSFORM,

        RandomVariableRescale(0.25, [(720, 480), (720, 576)]),
    
        RelativeCenterCrop([8, 8]),
    ])
    
    source_dataset = SingleImageFolder(root=r'F:\datasets\aug_test\src', 
        base_transform=base_transform, hr_transform=hr_transform, lr_transform=lr_transform)
    
    hr_folder = r'F:\datasets\aug_test\hr'
    lr_folder = r'F:\datasets\aug_test\lr'
    
    #Parallel(n_jobs=6)(delayed(process_j)(source_dataset, hr_folder, lr_folder, random.randint(0, len(source_dataset) -1), n=8) for j in range(2**13))
    for j in range(len(source_dataset)):
        (process_j)(source_dataset, hr_folder, lr_folder, j)
        

if __name__ == '__main__':
    main()