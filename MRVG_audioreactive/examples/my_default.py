import torch as th
import os
import numpy as np
import torch
import pickle
from PIL import Image
import sys
stylegan3_path = '/home/sinssinej7/private/self-research/Audio-reactive-video/maua_stylegan2/stylegan3'
sys.path.append(stylegan3_path)

import stylegan3.legacy as legacy
import stylegan3.dnnlib as dnnlib
import stylegan3
from stylegan3.torch_utils import misc
import audioreactive as ar

def load_generator(ckpt, dataparallel=False):
    """Loads a StyleGAN3 generator"""
    print('Loading StyleGAN3 generator from "%s"...' % ckpt)
    device = th.device('cuda')
    with dnnlib.util.open_url(ckpt) as f:
        generator = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    if dataparallel:
        generator = th.nn.DataParallel(generator)
    
    return generator

def initialize(args):
    args.lo_onsets = ar.onsets(args.audio, args.sr, args.n_frames, fmax=150, smooth=5, clip=97, power=2)
    args.hi_onsets = ar.onsets(args.audio, args.sr, args.n_frames, fmin=500, smooth=5, clip=99, power=2)
    return args

def smooth_signal_highlighted(signal, highlighted_factor=1.0,flag=True,method='highlighted',window_size=5):
    if not flag:
        print(signal)
        return signal
        
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]  
    elif signal.ndim > 2:
        signal = signal.reshape(signal.shape[0], -1)  

    
    if method =='highlighted':
        diff_signal = np.diff(signal, axis=0, prepend=signal[0:1])
        highlighted_signal = np.cumsum(diff_signal, axis=0)
        highlighted_signal = highlighted_factor * highlighted_signal
        highlighted_signal = highlighted_signal - np.min(highlighted_signal, axis=0)
        highlighted_signal = highlighted_signal / np.max(highlighted_signal, axis=0)
        print(highlighted_signal)


    return highlighted_signal.squeeze()  

def save_image(tensor, filename):
    # Change the shape of tensor to [height, width, channels] and scale to [0, 255]
    array = tensor.squeeze(0).clamp(0, 1).mul(255).permute(1, 2, 0).byte().cpu().numpy()
    Image.fromarray(array).save(filename)

def get_latents(selection, args):
    chroma = ar.chroma(args.audio, args.sr, args.n_frames,type='cens')
    with open(args.output_dir + '/chroma_cens.pkl', 'wb') as f:
        pickle.dump(chroma, f)
    # Apply highlighted to chroma with higher highlighted factor
    highlighted_factor = 1.0  
    chroma_highlighted = smooth_signal_highlighted(chroma, highlighted_factor=highlighted_factor)
    chroma_highlighted = th.tensor(chroma_highlighted).to("cuda") 
    ar.plot_spectra([chroma_highlighted],path=args.output_dir+'/chromagram_minmax_coolwarm.png', chroma=True)


    generator = load_generator('/home/sinssinej7/private/self-research/StyleGAN/stylegan3_train/00000-stylegan3-t-flower_data512-gpus2-batch16-gamma6.6/network-snapshot-002080.pkl')  # 적절한 체크포인트 경로로 변경하세요
    
    seeds = [64,64,6,6,49,49,6,6,5,5,10,4] # choose seeds

    for idx, i in enumerate(seeds):
        th.manual_seed(i)
        zs = th.randn((1, 512), device="cuda")
        saved.append(zs)
    zs = th.cat((saved), 0)
    
    custom_vectors = generator.mapping(zs, None).to("cuda")  
    chroma_latents = ar.chroma_weight_latents(chroma_highlighted, custom_vectors)

    latents = ar.gaussian_filter(chroma_latents, 4)
    
    lo_onsets = args.lo_onsets[:, None, None]
    hi_onsets = args.hi_onsets[:, None, None]
    
    lo_onsets = th.tensor(lo_onsets).to("cuda") 
    hi_onsets = th.tensor(hi_onsets).to("cuda")  

    latents = hi_onsets * custom_vectors[[-3]] + (1 - hi_onsets) * latents
    latents = lo_onsets * custom_vectors[[-9]] + (1 - lo_onsets) * latents
    
    latents = ar.gaussian_filter(latents, 2, causal=0.2)
    
    # os.makedirs('/home/sinssinej7/private/self-research/Audio-reactive-video/generated_images_wave', exist_ok=True)
    # for i, z in enumerate(zs):
    #     img = generator(z.unsqueeze(0), None, noise_mode='const', force_fp32=True)
    #     img = (img - img.min()) / (img.max() - img.min())  # 정규화

    #     save_image(img, f'/home/sinssinej7/private/self-research/Audio-reactive-video/generated_images_wave/image_seed_{seeds[i]}.png')

    return latents


