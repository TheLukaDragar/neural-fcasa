from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
import pickle as pkl

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as oc

from einops import rearrange
import numpy as np

import torch
from torch import nn
from torch.nn import functional as fn  # noqa

from einops.layers.torch import Rearrange
from torchaudio.transforms import InverseSpectrogram

from huggingface_hub import snapshot_download
from kornia.filters import MedianBlur

import soundfile as sf

from pathlib import Path

import sys

# Ensure the path to the project is in the PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


from neural_fcasa.utils.separator import main


@dataclass
class Context:
    model: nn.Module
    istft: nn.Module
    median_filt: nn.Module
    config: ListConfig | DictConfig


def add_common_args(parser):
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--out_ch", type=int, default=0)
    parser.add_argument("--medfilt_size", type=int, default=11)
    parser.add_argument("--noi_snr", type=float, default=None)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--dump_diar", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--batch_size", type=str, default=4, help="Batch size for processing"
    )
    parser.add_argument(
        "--chunk_len", type=int, default=10, help="Length of chunks in seconds"
    )


def initialize(args: Namespace, unk_args: list[str]):
    if args.model_path.startswith("hf://"):
        hf_path = str(args.model_path).removeprefix("hf://")
        args.model_path = Path(snapshot_download(hf_path))
    else:
        args.model_path = Path(args.model_path)

    config = oc.merge(
        oc.load(Path(args.model_path) / "config.yaml"),  # todo
        oc.from_cli(unk_args),
    )

    config.autocast = args.device.startswith("cuda")

    checkpoint_path = Path(args.model_path) / "version_0" / "checkpoints" / "last.ckpt"
    config.task._target_ += ".load_from_checkpoint"
    # print("config: ", config)
    model = instantiate(
        config.task,
        checkpoint_path=checkpoint_path,
        map_location=args.device,
    ).to(args.device)
    model.eval()

    # #check if we have a forward method
    # print("model: ", model)
    # #get type`
    # print("model type: ", type(model))
    # # print("model forward: ", model(torch.randn(1, 128)).shape)

    # #convert to torchscript
    # script = model.to_torchscript(method="trace")
    # torch.jit.save(script, "model_script.pt")

    istft = InverseSpectrogram(
        model.stft[0].n_fft, hop_length=model.stft[0].hop_length
    ).to(args.device)

    median_filt = nn.Sequential(
        Rearrange("b n t -> b n 1 t"),
        MedianBlur((1, args.medfilt_size)),
        Rearrange("b n 1 t -> b n t"),
    ).to(args.device)
    median_filt.eval()

    return Context(model, istft, median_filt, config)


def separate(
    src_filename: Path,
    dst_filename: Path,
    ctx: Context,
    args: Namespace,
    unk_args: list[str],
):
    model, istft = ctx.model, ctx.istft

    # Load wav file
    src_wav, sr = sf.read(src_filename, dtype=np.float32, always_2d=True)
    print("Length of src_wav (original): ", len(src_wav) / sr)

    #cut to 30s
    src_wav = src_wav[:110 * sr]


    # #pad the last chunk with zeros if necessary so its divisible by the chunk length
    # print("src_wav shape: ", src_wav.shape)
    # #src_wav shape:  (4801516, 8)

    # #pad so it divides by the chunk length
    # src_wav = np.pad(src_wav, ((0, args.chunk_len * sr - len(src_wav) % (args.chunk_len * sr)), (0, 0)), mode="constant")


    # print("src_wav shape after padding: ", src_wav.shape)


    # Assuming you have the necessary arguments and source waveform (src_wav)
    chunk_len = args.chunk_len  # Length of each chunk in seconds
    n_samples = len(src_wav)
    n_samples_per_chunk = chunk_len * sr

    print("n_samples: ", n_samples)
    print("n_samples_per_chunk: ", n_samples_per_chunk)

    # Calculate the number of full chunks with overlap
    n_full_chunks = (n_samples - n_samples_per_chunk) // (n_samples_per_chunk // 2) + 1

    # Create full chunks with overlap and convert to tensor
    full_chunks = [
        src_wav[i * (n_samples_per_chunk // 2) : i * (n_samples_per_chunk // 2) + n_samples_per_chunk]
        for i in range(n_full_chunks)
    ]
    full_chunks = np.stack(full_chunks)

    # Print the shape of the chunks to verify
    print("Shape of full_chunks: ", full_chunks.shape)
    


    # Convert full chunks to tensor and move to the model's device
    batches = torch.tensor(full_chunks, dtype=torch.float32).to(model.device)
    print("batches shape: ", batches.shape)

    # Batch size
    batch_size = args.batch_size

    all_channels_output = [[] for _ in range(ctx.config.n_mic)]
    diar_all = []

    for i in range(0, len(batches), batch_size):
        batch = batches[i : i + batch_size]

        print("processing batch: ", i//batch_size," shape: ", batch.shape)

        # src_wav = rearrange(torch.from_numpy(batch).to(model.device), "t m -> 1 m t")

        # torch.Size([8, 160000, 8])

        #reshaping to [B, M, T]
        src_wav = rearrange(batch, "b t m -> b m t")


        

        if src_wav.shape[1] != ctx.config.n_mic:
            raise RuntimeError(
                f"The number of input channels is {src_wav.shape[1]} but should be {ctx.config.n_mic}. "
                "Please specify a {ctx.config.n_mic}-channel signal to `src_filename`."
            )

        # calculate spectrogram
        xraw = model.stft(src_wav)[
            ..., : src_wav.shape[-1] // model.hop_length
        ]  # [B, F, M, T]
        scale = xraw.abs().square().clip(1e-6).mean(dim=(1, 2, 3), keepdims=True).sqrt()
        x = xraw / scale

        # encode
        z, w, g, Q, xt = model.encoder(x)
        w = ctx.median_filt(w).gt(args.thresh).to(torch.float32)
        print("w shape: ", w.shape)

        # decode
        lm = model.decoder(z)  # [B, F, N, T]

        # Wiener filtering
        yt = torch.einsum("bnt,bfnt,bfmn->bfmt", w, lm, g).add(1e-6)
        Qx_yt = torch.einsum("bfmn,bfnt->bfmt", Q, xraw) / yt
        # s = torch.einsum("bfm,bnt,bfnt,bfmn,bfmt->bnft", torch.linalg.inv(Q)[:, :, args.out_ch], w, lm, g, Qx_yt)

        # Loop through all channels and save them individually
        for ch in range(ctx.config.n_mic):
            s = torch.einsum(
                "bfm,bnt,bfnt,bfmn,bfmt->bnft",
                torch.linalg.inv(Q)[:, :, ch],
                w,
                lm,
                g,
                Qx_yt,
            )
            
            dst_wav = istft(s, src_wav.shape[-1])

            print("dst_wav shape: ", dst_wav.shape)

            # dst_wav = rearrange(dst_wav, "b m t -> t m")

            # if args.noi_snr is not None:
            #     scale = dst_wav.square().mean().sqrt().clip(1e-6) * 10 ** (
            #         -args.noi_snr / 20
            #     )
            #     dst_wav = dst_wav + torch.randn_like(dst_wav) * scale

            # if args.normalize:
            #     dst_wav /= dst_wav.abs().max().clip(1e-6)

            #torch.Size([8, 6, 160000])
            if args.noi_snr is not None:
                scale = dst_wav.square().mean(dim=(-2, -1), keepdim=True).sqrt().clip(1e-6) * 10 ** (-args.noi_snr / 20)
                dst_wav = dst_wav + torch.randn_like(dst_wav) * scale

            if args.normalize:
                dst_wav /= dst_wav.abs().max(dim=-1, keepdim=True).values.clip(1e-6)

            # Aggregate the results for each channel
            print("appending to all_channels_output: ", ch, dst_wav.shape)

            #check that we have full batches
            if src_wav.shape[0] != batch_size:
                print("padding the last batch with zeros")
                #pad the last batch with zeros
                #add extra batch entries to the last batch
                dst_wav = torch.cat([dst_wav, torch.zeros((batch_size - src_wav.shape[0], ctx.config.n_src, src_wav.shape[-1]), device=dst_wav.device)], dim=0)
                print("dst_wav shape after padding: ", dst_wav.shape)

            # 7 torch.Size([2, 6, 160000])
            #reshape dst_wav to remove batch dim 
            dst_wav = rearrange(dst_wav, "b m t -> m (b t)")

            
            print("appending to all_channels_output: ", ch, dst_wav.shape)

            all_channels_output[ch].append(dst_wav.cpu().numpy())
            
        diar_all.append(w.cpu().numpy())
      



                


    # Concatenate the results for all channels

    #pad the last batch with zeros if necessary in a

    print("all_channels_output: ", len(all_channels_output))
    print("diar_all: ", len(diar_all))


    #all_channels_output shape:  (8, 8, 6, 640000) reshae to (8, 6, 640000 * 8)
    all_channels_output = [np.concatenate(ch_output, axis=-1) for ch_output in all_channels_output]


    #stack the results
    all_channels_output = np.array(all_channels_output)
    diar_all = np.concatenate(diar_all, axis=0)

    print("all_channels_output shape: ", all_channels_output.shape)
    print("diar_all shape: ", diar_all.shape)

    from itertools import permutations


    final = [[] for _ in range(ctx.config.n_mic)]


    overlap_len = args.chunk_len * sr // 2

    for ch in range(ctx.config.n_mic):
        print("processing channel: ", ch)
        mic_output = all_channels_output[ch]
        print("mic_output shape: ", mic_output.shape) 


        perms_discovered = []

        #loop through the chunks of size chunk_len * sr
        for chunk in range(0, mic_output.shape[1], args.chunk_len * sr):
            print("processing chunk: ", chunk)

            current_chunk = mic_output[:, chunk:chunk + args.chunk_len * sr]
            # print("current_chunk shape: ", current_chunk.shape)

            current_chunk_overlap = current_chunk[:, overlap_len:]
            # print("current_chunk_overlap shape: ", current_chunk_overlap.shape)
            
            next_chunk = mic_output[:, chunk + args.chunk_len * sr:chunk + 2 * args.chunk_len * sr]

            # print("next_chunk shape: ", next_chunk.shape)

            next_chunk_overlap = next_chunk[:, :overlap_len]

            # print("next_chunk_overlap shape: ", next_chunk_overlap.shape)

            if next_chunk_overlap.shape[1] == 0:
                #append the last chunk
                continue

            #find the permutation that minimizes the difference between the current chunk and the next chunk
            min_diff = float("inf")
            min_perm = None

            perms = list(permutations(range(ctx.config.n_src)))

            # next_chunk shape:  (6, 160000)
            # next_chunk_overlap shape:  (6, 80000)


            min_diff = float('inf')
            min_perm = None

            for perm in perms:
                # Permute the first dimension of next_chunk_overlap
                next_chunk_perm = next_chunk_overlap[perm, :]

                # Calculate the difference
                diff = np.sum(np.abs(current_chunk_overlap - next_chunk_perm))

                if diff < min_diff:
                    min_diff = diff
                    min_perm = perm

            # print("min_diff: ", min_diff)

            if not min_perm == (0, 1, 2, 3, 4, 5):
                print("CHANGE DETECTED", min_perm)

            # print("min_perm: ", min_perm)
            perms_discovered.append(min_perm)


        print("perms_discovered: ", len(perms_discovered))

        #walk throut the chunks and apply the permutations
        final_mic_output = []

        first_chunk = mic_output[:, :overlap_len]
        final_mic_output.append(first_chunk)
        

        for i, perm in enumerate(perms_discovered):
            print("applying perm: ", perm)
            current_chunk = mic_output[:, i * args.chunk_len * sr + overlap_len:(i + 1) * args.chunk_len * sr]
            current_chunk_perm = current_chunk[perm, :]
            final_mic_output.append(current_chunk_perm)

        print("final_mic_output: ", len(final_mic_output))

        #concatenate the final mic output
        final_mic_output = np.array(final_mic_output)

        print("final_mic_output shape: ", final_mic_output.shape)
        final_mic_output =rearrange(final_mic_output, "a b c -> b (a c)")
        print("final_mic_output shape: ", final_mic_output.shape)

        final[ch] = final_mic_output


    for x in final:
        print("x final shape: ", x.shape)


    final = np.array(final)

    print("final shape: ", final.shape)


    all_channels_output = final



        
            
         

            





            




    #all_channels_output shape:  (240, 6, 160000)

    #reshape to 240 to 2 dims one is 30 aka length of the original signal and the other is 8 batch size

    #expand to 4 dims to 30, 8, 6, 160000
    # # all_channels_output = all_channels_output.reshape(full_chunks.shape[0], ctx.config.n_mic, all_channels_output.shape[1], all_channels_output.shape[2])

    

    print("all_channels_output shape: ", all_channels_output.shape)
    print("diar_all shape: ", diar_all.shape)

    #save the results
    for chunk in range(diar_all.shape[0]):
        print("processing chunk: ", chunk)
        chunk_diar = diar_all[chunk]

        #save the diar
        with open(dst_filename.with_name(f"{dst_filename.stem}_chunk{chunk}.diar"), "wb") as f:
            pkl.dump(chunk_diar, f)

    #save the audio
    for ch in range(ctx.config.n_mic):
        print("processing channel: ", ch)
        mic_output = all_channels_output[ch]

            #save the audio
        sf.write(dst_filename.with_name(f"{dst_filename.stem}_mic{ch}_{dst_filename.suffix}"), mic_output.T, sr, "PCM_24")
            
        






   



            

        #     # save separated signal for each channel
        #     channel_dst_filename = dst_filename.with_name(
        #         f"{dst_filename.stem}_ch{ch}{dst_filename.suffix}"
        #     )
        #     sf.write(channel_dst_filename, dst_wav.cpu().numpy(), sr, "PCM_24")

        # if args.dump_diar:
        #     with open(dst_filename.with_suffix(".diar"), "wb") as f:
        #         # if it exist delete

        #         print("dumping diar", f)
        #         pkl.dump(w.cpu().numpy(), f)

    # dst_wav = istft(s, src_wav.shape[-1])

    # if args.dump_diar:
    #     with open(dst_filename.with_suffix(".diar"), "wb") as f:
    #         pkl.dump(w.cpu().numpy(), f)

    # dst_wav = rearrange(dst_wav, "1 m t -> t m")

    # if args.noi_snr is not None:
    #     scale = dst_wav.square().mean().sqrt().clip(1e-6) * 10 ** (-args.noi_snr / 20)
    #     dst_wav = dst_wav + torch.randn_like(dst_wav) * scale

    # if args.normalize:
    #     dst_wav /= dst_wav.abs().max().clip(1e-6)

    # # save separated signal
    # sf.write(dst_filename, dst_wav.cpu().numpy(), sr, "PCM_24")


if __name__ == "__main__":
    main(add_common_args, initialize, separate)
