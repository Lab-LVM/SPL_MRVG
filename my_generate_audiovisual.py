import argparse
import gc
import os
import random
import time
import traceback
import uuid
import warnings

import librosa as rosa
import librosa.display
import numpy as np
import torch as th

import MRVG_audioreactive as ar
import generate
import my_render as render

import sys
import os

# Add the path to the StyleGAN3 directory
stylegan3_path = '/home/sinssinej7/private/self-research/Audio-reactive-video/maua_stylegan2/stylegan3'
sys.path.append(stylegan3_path)

import stylegan3.legacy as legacy
import stylegan3.dnnlib as dnnlib
import stylegan3
from stylegan3.torch_utils import misc



def load_generator(ckpt, dataparallel=False):
    """Loads a StyleGAN3 generator"""
    print('Loading StyleGAN3 generator from "%s"...' % ckpt)
    device = th.device('cuda')
    with dnnlib.util.open_url(ckpt) as f:
        generator = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    if dataparallel:
        generator = th.nn.DataParallel(generator)
    
    return generator


def generate(
    ckpt,
    audio_file,
    initialize=None,
    get_latents=None,
    get_noise=None,
    get_bends=None,
    get_rewrites=None,
    get_truncation=None,
    output_dir="./output",
    audioreactive_file="MRVG_MRVG_audioreactive/examples/my_default.py",
    offset=0,
    duration=-1,
    latent_file=None,
    shuffle_latents=False,
    G_res=1024,
    out_size=1024,
    fps=30,
    latent_count=12,
    batch=8,
    dataparallel=False,
    truncation=1.0,
    stylegan1=False,
    noconst=False,
    latent_dim=512,
    n_mlp=8,
    channel_multiplier=2,
    randomize_noise=False,
    ffmpeg_preset="slow",
    base_res_factor=1,
    output_file=None,
    args=None,
):
    # if args is empty (i.e. generate() called directly instead of through __main__)
    # create args Namespace with all locally available variables
    if args is None:
        kwargs = locals()
        args = argparse.Namespace()
        for k, v in kwargs.items():
            setattr(args, k, v)

    # ensures smoothing is independent of frame rate
    ar.set_SMF(args.fps / 30)

    time_taken = time.time()
    th.set_grad_enabled(False)

    audio, sr, duration = ar.load_audio(audio_file, offset, duration)

    args.audio = audio
    args.sr = sr

    n_frames = int(round(duration * fps))
    args.duration = duration
    args.n_frames = n_frames

    if initialize is not None:
        args = initialize(args)

    # ====================================================================================
    # =========================== generate audiovisual latents ===========================
    # ====================================================================================
    print("\ngenerating latents...")
    print('1')
    if get_latents is None:
        from MRVG_audioreactive.examples.my_default import get_latents
        print('1')

    if latent_file is not None:
        latent_selection = ar.load_latents(latent_file)
    else:
        #latent_count=12
        latent_selection = ar.generate_latents_stylegan3(
            latent_count, ckpt, latent_dim=512, truncation_psi=1.0
        )
    if shuffle_latents:
        random_indices = random.sample(range(len(latent_selection)), len(latent_selection))
        latent_selection = latent_selection[random_indices]
    np.save("workspace/last-latents.npy", latent_selection.numpy())

    latents = get_latents(selection=latent_selection, args=args) #chroma_weight_latents(chroma, latents)에 의해 base latents가 추출

    print(f"{list(latents.shape)} amplitude={latents.std()}\n")



    # ====================================================================================
    # ================ generate audiovisual network bending manipulations ================
    # ====================================================================================
    if get_bends is not None:
        print("generating network bends...")
        bends = get_bends(args=args)
    else:
        bends = []

    # ====================================================================================
    # ================ generate audiovisual model rewriting manipulations ================
    # ====================================================================================
    if get_rewrites is not None:
        print("generating model rewrites...")
        rewrites = get_rewrites(args=args)
    else:
        rewrites = {}

    # ====================================================================================
    # ========================== generate audiovisual truncation =========================
    # ====================================================================================
    if get_truncation is not None:
        print("generating truncation...")
        truncation = get_truncation(args=args)
    else:
        truncation = float(truncation)

    # ====================================================================================
    # ==== render the given (latent, noise, bends, rewrites, truncation) interpolation ===
    # ====================================================================================
    gc.collect()
    th.cuda.empty_cache()

    generator = load_generator(ckpt, dataparallel=False)

    print(f"\npreprocessing took {time.time() - time_taken:.2f}s\n")

    print(f"rendering {n_frames} frames...")
    if output_file is None:
        checkpoint_title = ckpt.split("/")[-1].split(".")[0].lower()
        track_title = audio_file.split("/")[-1].split(".")[0].lower()
        output_file = f"{output_dir}/{track_title}_{checkpoint_title}_{uuid.uuid4().hex[:8]}.mp4"
    render.render(
        generator=generator,
        latents=latents,
        audio_file=audio_file,
        offset=offset,
        duration=duration,
        batch_size=batch,
        truncation=truncation,
        bends=bends,
        rewrites=rewrites,
        out_size=out_size,
        output_file=output_file,
        randomize_noise=randomize_noise,
        ffmpeg_preset=ffmpeg_preset,
     )
    


    print(f"\ntotal time taken: {(time.time() - time_taken)/60:.2f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--audio_file", type=str)
    parser.add_argument("--audioreactive_file", type=str, default="MRVG_audioreactive/examples/my_default.py")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--offset", type=float, default=0)
    parser.add_argument("--duration", type=float, default=-1, help="length of rendered video in seconds")
    parser.add_argument("--latent_file", type=str, default=None)
    parser.add_argument("--shuffle_latents", action="store_true")
    parser.add_argument("--G_res", type=int, default=1024)
    parser.add_argument("--out_size", type=int, default=1024, help="rendered video size. Options: 512, 1024, 1920")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--latent_count", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dataparallel", action="store_true")
    parser.add_argument("--truncation", type=float, default=1.0)
    parser.add_argument("--stylegan1", action="store_true")
    parser.add_argument("--noconst", action="store_true")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--randomize_noise", action="store_true")
    parser.add_argument("--base_res_factor", type=float, default=1)
    parser.add_argument("--ffmpeg_preset", type=str, default="slow")
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    # ensure output_dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    # transform file path to python module string
    modnames = args.audioreactive_file.replace(".py", "").replace("/", ".").split(".")

    # try to load each of the standard functions from the specified file
    func_names = ["initialize", "get_latents", "get_noise", "get_bends", "get_rewrites", "get_truncation"]
    funcs = {}
    for func in func_names:
        try:
            file = __import__(".".join(modnames[:-1]), fromlist=[modnames[-1]]).__dict__[modnames[-1]]
            funcs[func] = getattr(file, func)
        except AttributeError as error:
            print(f"No '{func}' function found in --audioreactive_file, using default...")
            funcs[func] = None
        except:
            if funcs.get(func, "error") == "error":
                print("Error while loading --audioreactive_file...")
                traceback.print_exc()
                exit(1)

    # override with args from the OVERRIDE dict in the specified file
    arg_dict = vars(args).copy()
    try:
        file = __import__(".".join(modnames[:-1]), fromlist=[modnames[-1]]).__dict__[modnames[-1]]
        for arg, val in getattr(file, "OVERRIDE").items():
            arg_dict[arg] = val
            setattr(args, arg, val)
    except:
        pass  # no overrides, just continue

    ckpt = arg_dict.pop("ckpt", None)
    audio_file = arg_dict.pop("audio_file", None)

    # splat kwargs to function call
    # (generate() has all kwarg defaults specified again to make it amenable to ipynb usage)
    generate(ckpt=ckpt, audio_file=audio_file, **funcs, **arg_dict, args=args)
