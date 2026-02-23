import queue
from threading import Thread

import ffmpeg
import numpy as np
import PIL.Image
import torch as th
from tqdm import tqdm

th.set_grad_enabled(False)
th.backends.cudnn.benchmark = True

def render(
    generator,
    latents,  
    offset,
    duration,
    batch_size,  
    out_size,
    output_file,
    audio_file=None,
    truncation=1.0,
    bends=[],
    rewrites={},
    randomize_noise=False,
    ffmpeg_preset="slow",
):
    split_queue = queue.Queue()
    render_queue = queue.Queue()

    # postprocesses batched torch tensors to individual RGB numpy arrays
    def split_batches(jobs_in, jobs_out):
        while True:
            try:
                imgs = jobs_in.get(timeout=5)
            except queue.Empty:
                return
            imgs = (imgs.clamp_(-1, 1) + 1) * 127.5  
            imgs = imgs.permute(0, 2, 3, 1) 
            for img in imgs:
                jobs_out.put(img.cpu().numpy().astype(np.uint8)) 
            jobs_in.task_done()

    # start background ffmpeg process that listens on stdin for frame data
    if out_size == 512:
        output_size = "512x512"
    elif out_size == 1024:
        output_size = "1024x1024"
    elif out_size == 1920:
        output_size = "1920x1080"
    elif out_size == 1080:
        output_size = "1080x1920"
    else:
        raise Exception("The only output sizes currently supported are: 512, 1024, 1080, or 1920")

    if audio_file is not None:
        audio = ffmpeg.input(audio_file, ss=offset, t=duration, guess_layout_max=0)
        video = (
            ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", framerate=len(latents) / duration, s=output_size)
            .output(
                audio,
                output_file,
                framerate=len(latents) / duration,
                vcodec="libx264",
                pix_fmt="yuv420p",
                preset=ffmpeg_preset,
                audio_bitrate="320K",
                ac=2,
                v="warning",
            )
            .global_args("-hide_banner")
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
    else:
        video = (
            ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", framerate=len(latents) / duration, s=output_size)
            .output(
                output_file,
                framerate=len(latents) / duration,
                vcodec="libx264",
                pix_fmt="yuv420p",
                preset=ffmpeg_preset,
                v="warning",
            )
            .global_args("-hide_banner")
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    # writes numpy frames to ffmpeg stdin as raw rgb24 bytes
    def make_video(jobs_in):
        w, h = [int(dim) for dim in output_size.split("x")]
        for _ in tqdm(range(len(latents)), position=0, leave=True, ncols=80):
            img = jobs_in.get(timeout=5)
            if img.shape[1] == 2048:
                img = img[:, 112:-112, :]  # 2048x2048 이미지를 1920x1080으로 크롭 및 리사이즈
                im = PIL.Image.fromarray(img)
                img = np.array(im.resize((1920, 1080), PIL.Image.BILINEAR))
            elif img.shape[0] == 2048:
                img = img[112:-112, :, :]  # 2048x2048 이미지를 1080x1920으로 크롭 및 리사이즈
                im = PIL.Image.fromarray(img)
                img = np.array(im.resize((1080, 1920), PIL.Image.BILINEAR))
            assert (
                img.shape[1] == w and img.shape[0] == h
            ), f"""generator's output image size does not match specified output size: \n
                got: {img.shape[1]}x{img.shape[0]}\t\tshould be {output_size}"""
            video.stdin.write(img.tobytes())  # ffmpeg에 이미지 데이터를 전달
            jobs_in.task_done()
        video.stdin.close()
        video.wait()

    splitter = Thread(target=split_batches, args=(split_queue, render_queue))
    splitter.daemon = True
    renderer = Thread(target=make_video, args=(render_queue,))
    renderer.daemon = True

    # make all data that needs to be loaded to the GPU float, contiguous, and pinned
    # the entire process is severely memory-transfer bound, but at least this might help a little
    if not latents.is_cuda:
        latents = latents.float().contiguous().pin_memory()  

    param_dict = dict(generator.named_parameters())
    original_weights = {}
    for param, (rewrite, modulation) in rewrites.items():
        rewrites[param] = [rewrite, modulation.float().contiguous().pin_memory()]
        original_weights[param] = param_dict[param].copy().cpu().float().contiguous().pin_memory()

    for bend in bends:
        if "modulation" in bend:
            bend["modulation"] = bend["modulation"].float().contiguous().pin_memory()

    if not isinstance(truncation, float):
        truncation = truncation.float().contiguous().pin_memory()

    # 배치 단위로 이미지 생성
    for n in range(0, len(latents), batch_size):
        # 현재 배치를 GPU로 전송
        latent_batch = latents[n : n + batch_size].cuda(non_blocking=True)

        bend_batch = []
        if bends is not None:
            for bend in bends:
                if "modulation" in bend:
                    transform = bend["transform"](bend["modulation"][n : n + batch_size].cuda(non_blocking=True))
                    bend_batch.append({"layer": bend["layer"], "transform": transform})
                else:
                    bend_batch.append({"layer": bend["layer"], "transform": bend["transform"]})

        for param, (rewrite, modulation) in rewrites.items():
            transform = rewrite(modulation[n : n + batch_size])
            rewritten_weight = transform(original_weights[param]).cuda(non_blocking=True)
            param_attrs = param.split(".")
            mod = generator
            for attr in param_attrs[:-1]:
                mod = getattr(mod, attr)
            setattr(mod, param_attrs[-1], th.nn.Parameter(rewritten_weight))

        if not isinstance(truncation, float):
            truncation_batch = truncation[n : n + batch_size].cuda(non_blocking=True)
        else:
            truncation_batch = truncation

        # generator를 통해 이미지를 생성 (W 벡터를 입력으로 사용)
        outputs = generator.synthesis(ws=latent_batch, noise_mode='const')  # StyleGAN3에서 ws는 latent_batch

        # 생성된 이미지를 큐에 추가하여 ffmpeg로 전송
        split_queue.put(outputs)

        if n == 0:
            splitter.start()
            renderer.start()

    splitter.join()
    renderer.join()


def write_video(arr, output_file, fps):
    print(f"writing {arr.shape[0]} frames...")

    output_size = "x".join(reversed([str(s) for s in arr.shape[1:-1]]))

    ffmpeg_proc = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", framerate=fps, s=output_size)
        .output(output_file, framerate=fps, vcodec="libx264", preset="slow", v="warning")
        .global_args("-benchmark", "-stats", "-hide_banner")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for frame in arr:
        ffmpeg_proc.stdin.write(frame.astype(np.uint8).tobytes())

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()


# Example usage
# generator = load_generator('path/to/your/stylegan3-network.pkl', dataparallel=False)
# latents = generate_latents_stylegan3(n_latents, 'path/to/your/stylegan3-network.pkl', latent_dim)
# render(
#     generator=generator,
#     latents=latents,
#     offset=0,
#     duration=10,
#     batch_size=8,
#     out_size=1024,
#     output_file='output.mp4',
#     audio_file='audio.wav',
#     truncation=0.7,
#     bends=[],
#     rewrites={},
#     randomize_noise=False,
#     ffmpeg_preset="slow",
# )
