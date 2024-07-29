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


import pickle
import einops
import torch
import numpy as np


import soundfile as sf

# open audio file .wav
import torch
from copy import deepcopy
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)


# Ensure the path to the project is in the PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


from neural_fcasa.utils.separator import main

import plotly.express as px

import pandas as pd
import umap
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform


def calculate_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

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
    # src_wav = src_wav[:110 * sr]


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


    stepsize = n_samples_per_chunk
    #aproximapltey 0.5s overlap

    # Calculate the number of full chunks with overlap
    n_full_chunks = (n_samples - n_samples_per_chunk) // stepsize + 1

    # Create full chunks with overlap and convert to tensor
    # full_chunks = [
    #     src_wav[i * (n_samples_per_chunk // 2) : i * (n_samples_per_chunk // 2) + n_samples_per_chunk]
    #     for i in range(n_full_chunks)
    # ]

    full_chunks = [
        src_wav[i * stepsize : i * stepsize + n_samples_per_chunk]
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
    diar_all_raw = []
    z_all = []

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

        # save z
        z_all.append(z.cpu().numpy())


        w_raw = w.clone().to(torch.float32)


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

            # padding_time = 0
            # #check that we have full batches
            # if src_wav.shape[0] != batch_size:
            #     print("padding the last batch with zeros")
            #     #pad the last batch with zeros
            #     #add extra batch entries to the last batch
            #     padding_time = (batch_size - src_wav.shape[0]) * src_wav.shape[-1]
            #     print("padding_time: ", padding_time)
            #     dst_wav = torch.cat([dst_wav, torch.zeros((batch_size - src_wav.shape[0], ctx.config.n_src, src_wav.shape[-1]), device=dst_wav.device)], dim=0)
            #     print("dst_wav shape after padding: ", dst_wav.shape)

            # 7 torch.Size([2, 6, 160000])
            #reshape dst_wav to remove batch dim 
            dst_wav = rearrange(dst_wav, "b m t -> m (b t)")
            #remove the padded entries
            # dst_wav = dst_wav[:, :-padding_time]

            

            
            print("appending to all_channels_output: ", ch, dst_wav.shape)

            all_channels_output[ch].append(dst_wav.cpu().numpy())
            
        diar_all.append(w.cpu().numpy())
        diar_all_raw.append(w_raw.cpu().numpy())
        
      



                


    # Concatenate the results for all channels

    #pad the last batch with zeros if necessary in a

    print("all_channels_output: ", len(all_channels_output))
    print("diar_all: ", len(diar_all))
    print("diar_all_raw: ", len(diar_all_raw))
    print("z_all: ", len(z_all))


    #all_channels_output shape:  (8, 8, 6, 640000) reshae to (8, 6, 640000 * 8)
    all_channels_output = [np.concatenate(ch_output, axis=-1) for ch_output in all_channels_output]


    #stack the results
    all_channels_output = np.array(all_channels_output)
    diar_all = np.concatenate(diar_all, axis=0)
    diar_all_raw = np.concatenate(diar_all_raw, axis=0)
    z_all = np.concatenate(z_all, axis=0)

    print("all_channels_output shape: ", all_channels_output.shape)
    print("diar_all shape: ", diar_all.shape)

    # Extract the number of chunks from diar_all
    n_chunks = diar_all.shape[0]

    # Determine the shape for the final tensor
    n_mic = all_channels_output.shape[0]
    n_sources = all_channels_output.shape[1]
    chunk_len_samples = chunk_len * sr

    # Verify that all_channels_output has enough samples
    expected_samples = n_chunks * chunk_len_samples
    assert all_channels_output.shape[-1] >= expected_samples, "Not enough samples in all_channels_output"

    # Reshape the all_channels_output to the final shape directly
    final_output = all_channels_output[:, :, :expected_samples].reshape(n_mic, n_sources, n_chunks, chunk_len_samples)
    final_output = np.moveaxis(final_output, 2, 1)  # Change axis to (n_mic, n_chunks, n_sources, chunk_len_samples)

    print("final_output shape: ", final_output.shape)
    # final_output shape:  (8, 21, 6, 160000)

    #pickle the results
    with open(dst_filename.with_name(f"{dst_filename.stem}_final_output.pkl"), "wb") as f:
        pkl.dump(final_output, f)

    with open(dst_filename.with_name(f"{dst_filename.stem}_diar_all.pkl"), "wb") as f:
        pkl.dump(diar_all, f)

    with open(dst_filename.with_name(f"{dst_filename.stem}_diar_all_raw.pkl"), "wb") as f:
        pkl.dump(diar_all_raw, f)

    with open(dst_filename.with_name(f"{dst_filename.stem}_z_all.pkl"), "wb") as f:
        pkl.dump(z_all, f)


    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ArtifactHook
    from pyannote.core import Segment, Annotation
    from pyannote.audio.pipelines.utils.hook import ArtifactHook, ProgressHook
    from typing import Any, Mapping, Optional, Text
    import plotly.graph_objects as go

    from pyannote.core import Segment, Annotation
    from pyannote.audio.core.io import Audio




    class CombinedHook:
        """Composite Hook to save artifacts and show progress of each internal step.

        Parameters
        ----------
        artifacts: list of str, optional
            List of steps to save. Defaults to all steps.
        file_key: str, optional
            Key used to store artifacts in `file`.
            Defaults to "artifact".
        transient: bool, optional
            Clear the progress on exit. Defaults to False.

        Usage
        -----
        >>> with CombinedHook() as hook:
        ...     output = pipeline(file, hook=hook)
        # file["artifact"] contains a dict with artifacts of each step
        """

        def __init__(
            self, *artifacts, file_key: str = "artifact", transient: bool = False
        ):
            self.artifact_hook = ArtifactHook(*artifacts, file_key=file_key)
            self.progress_hook = ProgressHook(transient=transient)

        def __enter__(self):
            self.artifact_hook.__enter__()
            self.progress_hook.__enter__()
            return self

        def __exit__(self, *args):
            self.artifact_hook.__exit__(*args)
            self.progress_hook.__exit__(*args)

        def __call__(
            self,
            step_name: Text,
            step_artifact: Any,
            file: Optional[Mapping] = None,
            total: Optional[int] = None,
            completed: Optional[int] = None,
        ):
            self.artifact_hook(step_name, step_artifact, file, total, completed)
            self.progress_hook(step_name, step_artifact, file, total, completed)
    

    
    audio, sr = sf.read(src_filename, dtype=np.float32, always_2d=True)


    print(audio.shape, sr)

    # (4801516, 8)

    # switch dims
    audio = einops.rearrange(audio, "n c -> c n")
    # (8, 4801516)

    # convert to float 32
    audio = audio.astype(np.float32)

    all_full_embedings = []
    all_hard_clusters = []
    all_diars = []
    all_centroids = []

    # perform speaker diarization on full audio
    pipeline = Pipeline.from_pretrained(
        "./config2.yaml", use_auth_token="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"
    )

    pipeline.to(torch.device("cuda"))

    for i in range(audio.shape[0]):
        current_microphone = audio[i]

        # audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}
        # type(waveform)=<class 'torch.Tensor'>
        # waveform.shape=torch.Size([1, 480000])
        # waveform.dtype=torch.float32

        audio_in_memory = {
            "waveform": torch.from_numpy(current_microphone).unsqueeze(0),
            "sample_rate": 16000,
        }

        # run the pipeline on an audio file

        with CombinedHook() as hook:

            diarization, embedings = pipeline(
                audio_in_memory, hook=hook, return_embeddings=True
            )

        print(audio_in_memory.keys())

        # full_embedings = audio_in_memory["artifact"][
        #     "embeddings"
        # ]  # (num_chunks, local_num_speakers, dimension)
        # hard_clusters = audio_in_memory["artifact"][
        #     "hard_clusters"
        # ]  # (num_chunks, local_num_speakers)

        # all_full_embedings.append(full_embedings)
        # all_hard_clusters.append(hard_clusters)
        all_diars.append(diarization)
        all_centroids.append(embedings)


    #save to pkl
    with open(dst_filename.with_name(f"{dst_filename.stem}_pyannotate_diars.pkl"), "wb") as f:
        pkl.dump(all_diars, f)

    with open(dst_filename.with_name(f"{dst_filename.stem}_pyannotate_centroids.pkl"), "wb") as f:
        pkl.dump(all_centroids, f)

        

    
    noise = final_output[:,:,5,:]
    print(noise.shape)

    noise = einops.rearrange(noise, "m c t -> m (c t)")

    all_max_amplitude = []
    all_snrs = []

    # Use the same averaged noise for all speakers
    global_noise = np.mean(noise, axis=0)

    print(global_noise.shape)

    for i in range(audio.shape[0]):

        sr = 16000

        current_diar = all_diars[i]

        current_mic = audio[i]

        current_speaker_max_amplitude = {}
        current_speaker_noise_segments = {}
        current_speaker_signal_segments = {}

        for turn, _, speaker in current_diar.itertracks(yield_label=True):

            start = int(turn.start * sr)
            end = int(turn.end * sr)

            # Check if both start and end are over limit
            if start > current_mic.shape[0] and end > current_mic.shape[0]:
                print("both start and end over limit, skipping")
                continue

            # Check if end is over limit
            if end > current_mic.shape[0]:
                print("end over limit setting to end of file", current_mic.shape[0])
                end = current_mic.shape[0]

            # Check if start and end are the same
            if start == end:
                print("start and end are the same, skipping")
                continue

            segment = current_mic[start:end]
            segment = np.array(segment)

            noise_segment = global_noise[start:end]

            max_amplitude = np.max(np.abs(segment))

            if speaker not in current_speaker_max_amplitude:
                current_speaker_max_amplitude[speaker] = 0
                current_speaker_signal_segments[speaker] = []

            current_speaker_max_amplitude[speaker] = max(
                max_amplitude, current_speaker_max_amplitude[speaker]
            )

            current_speaker_signal_segments[speaker].append(segment)


            if speaker not in current_speaker_noise_segments:
                current_speaker_noise_segments[speaker] = []
            
            current_speaker_noise_segments[speaker].append(noise_segment)



        current_speaker_snr = {}
        for speaker in current_speaker_signal_segments:
            accumulated_signal = np.concatenate(current_speaker_signal_segments[speaker])
            accumulated_noise = np.concatenate(current_speaker_noise_segments[speaker])

            snr = calculate_snr(accumulated_signal, accumulated_noise)
            current_speaker_snr[speaker] = snr

        current_speaker_max_amplitude = dict(
            sorted(
                current_speaker_max_amplitude.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        current_speaker_snr = dict(
            sorted(
                current_speaker_snr.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        all_max_amplitude.append(current_speaker_max_amplitude)
        all_snrs.append(current_speaker_snr)

    print(all_max_amplitude)
    print(all_snrs)

    # Plot histograms for each run
    fig = go.Figure()

    for i in range(len(all_max_amplitude)):
        current_max_amplitude = all_max_amplitude[i]
        for speaker, amplitude in current_max_amplitude.items():
            fig.add_trace(go.Bar(x=[f"mic_{i}_{speaker}"], y=[amplitude], name=f"Amplitude mic_{i}_{speaker}"))

    fig.update_layout(yaxis_title="Amplitude", xaxis_title="Speaker", title="Speaker Amplitude")
    fig.show()

    # Plot SNR for each run
    fig_snr = go.Figure()

    for i in range(len(all_snrs)):
        current_snr = all_snrs[i]
        for speaker, snr in current_snr.items():
            fig_snr.add_trace(go.Bar(x=[f"mic_{i}_{speaker}"], y=[snr], name=f"SNR mic_{i}_{speaker}"))

    fig_snr.update_layout(yaxis_title="SNR (dB)", xaxis_title="Speaker", title="Speaker SNR")
    fig_snr.show()

    best_speakers_per_mic = []
    for i in range(len(all_max_amplitude)):
        current_max_amplitude = all_max_amplitude[i]
        # Check if there are any speakers
        if len(current_max_amplitude) == 0:
            best_speakers_per_mic.append("no speakers")
        else:
            best_speakers_per_mic.append(list(current_max_amplitude.keys())[0])

    print("max amplitude speaker per mic: ", best_speakers_per_mic)

    best_snr_per_mic = []
    for i in range(len(all_snrs)):
        current_snr = all_snrs[i]
        # Check if there are any speakers
        if len(current_snr) == 0:
            best_snr_per_mic.append("no speakers")
        else:
            best_snr_per_mic.append(list(current_snr.keys())[0])

    print("best snr speaker per mic: ", best_snr_per_mic)
    # Assuming all_diars and all_max_amplitude are already defined
    # Load diarization data

    #plot all centroids on umap

    # centroid_speaker_labels_run = []

    # runsss= []
    # sizes  = []

    # for i in range(len(all_centroids)):

    #     current_centroids = all_centroids[i]

    #     #sorted by speaker automatically
    #     for j in range(current_centroids.shape[0]):

    #         runsss.append(i)


    #         speaker_key = f"SPEAKER_{j:02d}"
    #         centroid_speaker_labels_run.append(f"mic_{i}_speaker_{speaker_key}")
    #         sizes.append(all_max_amplitude[i].get(speaker_key, 2222))  # Default size to 1 if key not found

    #         # sizes.append(all_max_amplitude[i][f"SPEAKER_0{j}"])

    # print(centroid_speaker_labels_run)

    # all_centroids_flat = np.concatenate(all_centroids,axis=0)

    # print(all_centroids_flat.shape)

    

    # reducer = umap.UMAP(metric="cosine",n_components=3)

    # umap_centroids = reducer.fit_transform(all_centroids_flat)

    # #plotly

   

    # df = pd.DataFrame(umap_centroids, columns=["x", "y","z"])
    # #
    # df["speaker"] = centroid_speaker_labels_run

    # #to string 

    # df["run"] = runsss

    # df["run"] = df["run"].apply(lambda x: "run_"+str(x))    


    # #sizez proportianal to max amplitude of that speaket 

    # df["size"] = sizes



    # fig = px.scatter_3d(df, x="x", y="y", z="z", color="speaker",size="size",hover_data=["run"])



    # fig.show()


    # fig = px.scatter_3d(df, x="x", y="y", z="z", color="run")
    # fig.show()



    # # Concatenate all centroids
    # all_centroids_flat = np.concatenate(all_centroids, axis=0)
    # print(all_centroids_flat.shape)

    # #set nan values to 0
    # all_centroids_flat = np.nan_to_num(all_centroids_flat)

    # #  Check if there are any NaN values left
    # if np.isnan(all_centroids_flat).any() or np.isinf(all_centroids_flat).any():
    #     raise ValueError("Centroids contain NaNs or infinite values.")
    
    # # Check for zero vectors and handle them
    # zero_vectors = np.all(all_centroids_flat == 0, axis=1)
    # if np.any(zero_vectors):
    #     print(f"Found {np.sum(zero_vectors)} zero vectors. Handling them.")
    #     # Add small noise to zero vectors to avoid division by zero
    #     all_centroids_flat[zero_vectors] += np.random.normal(0, 1e-10, all_centroids_flat.shape[1])


    # # Perform agglomerative clustering using cosine distance
    # threshold = 0.7045654963945799
    # distance_matrix = squareform(pdist(all_centroids_flat, metric='cosine'))

    # if np.isnan(distance_matrix).any() or np.isinf(distance_matrix).any():
        
    #     raise ValueError("Distance matrix contains NaNs or infinite values.")
    

    

    # agg_clustering = AgglomerativeClustering(
    #     n_clusters=None, linkage='average', distance_threshold=threshold
    # )
    # cluster_labels = agg_clustering.fit_predict(distance_matrix)

    # # Perform UMAP on centroids
    # reducer = umap.UMAP(metric="cosine", n_components=3)
    # umap_centroids = reducer.fit_transform(all_centroids_flat)

    # # Prepare data for plotting
    # centroid_speaker_labels_run = []
    # runsss = []
    # sizes = []

    # for i in range(len(all_centroids)):
    #     current_centroids = all_centroids[i]
    #     for j in range(current_centroids.shape[0]):
    #         runsss.append(i)
    #         speaker_key = f"SPEAKER_{j:02d}"
    #         centroid_speaker_labels_run.append(f"mic_{i}_{speaker_key}")
            
    #         sizes.append(all_max_amplitude[i].get(speaker_key, 2222))

    # # Create DataFrame
    # df = pd.DataFrame(umap_centroids, columns=["x", "y", "z"])
    # df["speaker"] = centroid_speaker_labels_run
    # df["run"] = ["run_" + str(x) for x in runsss]
    # df["size"] = sizes
    # df["cluster"] = cluster_labels

    # # Plot the results
    # fig = px.scatter_3d(df, x="x", y="y", z="z", color="cluster", hover_data=["run"])
    # fig.show()

    # fig_run = px.scatter_3d(df, x="x", y="y", z="z", color="run")
    # fig_run.show()

    # print("Clusters correspond to unique speakers across different runs.")

    # # Print number of discovered speakers
    # num_discovered_speakers = len(np.unique(cluster_labels))
    # print("Number of discovered speakers:", num_discovered_speakers)

    # # Find the loudest speaker per discovered cluster in each microphone
    # loudest_speakers = {}

    # for cluster in np.unique(cluster_labels):
    #     loudest_speakers[cluster] = {}
    #     cluster_indices = np.where(cluster_labels == cluster)[0]
    #     max_amplitude = -1
    #     loudest_speaker = None
    #     loudest_run = None
    #     for idx in cluster_indices:
    #         run = runsss[idx]
    #         speaker_label = df["speaker"].iloc[idx]
    #         speaker_key = f"SPEAKER_{int(speaker_label.split('_')[-1]):02d}"
    #         amplitude = all_max_amplitude[run].get(speaker_key, -1)
    #         if amplitude > max_amplitude:
    #             max_amplitude = amplitude
    #             loudest_speaker = speaker_label
    #             loudest_run = run
    #     loudest_speakers[cluster] = {f"run_{loudest_run}": loudest_speaker}

    # # Print loudest speakers for each discovered cluster in each microphone
    # for cluster, speakers in loudest_speakers.items():
    #     print(f"Cluster {cluster}:")
    #     for run, speaker in speakers.items():
    #         print(f"  {run}: {speaker}")

    # # Create an array to store the representative label for each run
    # representative_labels = []

    # for run in range(len(all_centroids)):
    #     assigned = False
    #     for cluster, speakers in loudest_speakers.items():
    #         if f"run_{run}" in speakers:
    #             representative_labels.append(speakers[f"run_{run}"])
    #             assigned = True
    #             break
    #     if not assigned:
    #         # Assign the speaker with max amplitude if no discovered speaker is found
    #         max_amplitude = -1
    #         loudest_speaker = None
    #         for j in range(all_centroids[run].shape[0]):
    #             speaker_key = f"SPEAKER_{j:02d}"
    #             amplitude = all_max_amplitude[run].get(speaker_key, -1)
    #             if amplitude > max_amplitude:
    #                 max_amplitude = amplitude
    #                 loudest_speaker = f"mic_{run}_{speaker_key}"
    #         representative_labels.append(loudest_speaker)

    # # Print the representative labels for each run
    # print("Representative labels for each run:")
    # for run, label in enumerate(representative_labels):
    #     print(f"Run(mic) {run}: {label}")



    #get representitive lables from the best snr
    representative_labels = [f"mic_{i}_{best_snr_per_mic[i]}" if best_snr_per_mic[i] != "no speakers" else None for i in range(len(best_snr_per_mic))]
    print("Representative labels for each run:")
    for run, label in enumerate(representative_labels):
        print(f"Run(mic) {run}: {label}")





    fig = go.Figure()

    # Collect all unique labels for the y-axis
    unique_labels = []

    for i in range(len(all_diars)):
        current_diar = all_diars[i]
        for speaker in current_diar.labels():
            unique_label = f"mic_{i}_{speaker}"
            unique_labels.append(unique_label)

    # Remove duplicates and sort labels
    unique_labels = sorted(set(unique_labels))

    for i in range(len(all_diars)):
        current_diar = all_diars[i]

        #get representitive speakers for each run
        current_max_amplitude = all_max_amplitude[i]
        


        run_color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]

        for turn, _, speaker in current_diar.itertracks(yield_label=True):
            y_label = f"mic_{i}_{speaker}"

            #this is flat signal with stacked source separatin channels 

            start = turn.start
            end = turn.end

        
            
            # if y_label == "mic_2_SPEAKER_03":
            #     print(start,end,turn.start,turn.end)


            #calculate max amplitude of this segment


        

            #energy values are rea

            #use plotly colors scale to get color if representitive speaker

            #check if the speaker is the representitive speaker
            #fin the index if the curent speaker in current_max_amplitude
            speaker_index = list(current_max_amplitude.keys()).index(speaker)
            # print(speaker_index)

            #get max amplitude of the speaker

            #get color based on index  use plotly colors make the first one really hot
            c = px.colors.sequential.Reds[(8-speaker_index)]

            #check if the speaker is the representitive speaker
            print(representative_labels[i],y_label)
            if y_label == representative_labels[i]:

                #color based on index of representitive speaker
                c = px.colors.sequential.Greens[(8-representative_labels.index(y_label))]

            

            fig.add_trace(go.Scatter(
                x=[start,end],
                y=[y_label, y_label],
                mode='lines',
                line=dict(color=c, width=10),
                name=y_label,
                legendgroup=f"mic_{i}",
                showlegend=(y_label not in [trace.name for trace in fig.data])
            ))

    # Update layout to set y-axis as category type and use the unique labels
    fig.update_layout(
        yaxis=dict(
            title='Speakers',
            tickmode='array',
            tickvals=unique_labels,
            ticktext=unique_labels,
            categoryorder='array',
            categoryarray=unique_labels
        ),
        xaxis=dict(title='Time'),
        title='Diarization Visualization',
        legend=dict(title='Speakers', itemsizing='constant')
    )

    fig.show()

   
    audio_duration = int(audio.shape[1] / sr)

    # Process diarization data
    num_chunks = final_output.shape[1]

    merged_annotation_representative_speakers = Annotation()

    # mics_to_keep = [0, 1, 3, 5]

    mics_to_keep = list(range(audio.shape[0]))

    print("Mics to keep: ", mics_to_keep)


    for i in range(len(all_diars)):
        
        print(i)
    
        #representitive speaker
        representitive_speaker = representative_labels[i]
    

        #check if no speaker
        if representitive_speaker == None:
            filtered_annotation = all_diars[i].copy()
            print("No speaker detected",filtered_annotation.labels())
            print("REMOVING MIC ",i)
            #remove mic from mic to keep
            mics_to_keep = [x for x in mics_to_keep if x != i]

            print("Mics to keep: ", mics_to_keep)


            continue

            

        mic, speaker = representitive_speaker.split("_SPEAKER_")
        print(mic,speaker)
        mic = int(mic.split("mic_")[1])
        speaker = f"SPEAKER_{int(speaker):02d}"

        filtered_annotation = all_diars[i].copy().subset([speaker])
        # Update the merged annotation with the filtered annotation
        filtered_annotation = filtered_annotation.rename_labels(
            mapping={speaker: f"rep_spk_mic_{mic}_{speaker}"} 
        )

        merged_annotation_representative_speakers.update(filtered_annotation)

    overlaps = merged_annotation_representative_speakers.get_overlap()

    # loop throught the merged annotation and add the segments to the plot each speaker discratized in different y

    # Load the diarization data
    diar_data_raw = diar_all_raw.copy()

    #remove the diarization of noise since its always 1 its in postition5

    diar_data_raw = diar_data_raw[:, :5, :]
    print(diar_data_raw.shape)



    discratized_segments = merged_annotation_representative_speakers.discretize(support=Segment(0, audio_duration))

    # Assuming discratized_segments is already defined and loaded
    np_dicratized = discratized_segments.data.copy()

    print(np_dicratized.shape)

    # Reshape np_dicratized
    np_dicratized = einops.rearrange(np_dicratized, '(a b) c -> a c b', a=num_chunks, b=1000)



    def find_best_assigment(raw_chunk, discratized_chunk):
        # Calculate the pairwise distances between the raw and discratized chunks
        print(raw_chunk.shape)
        print(discratized_chunk.shape)

        #(6, 1000)
        #(4, 1000)

        #ssign each 4 an element of 6
        assigments = []
        for i in range(discratized_chunk.shape[0]):
            best = 0
            lowest_error = np.inf
            for j in range(raw_chunk.shape[0]):
                error = np.sum(np.abs(raw_chunk[j] - discratized_chunk[i]))
                print(f"Error for {i} and {j}: {error}")
                if error < lowest_error:
                    lowest_error = error

                    best = j

            assigments.append(best)

        return np.array(assigments)

        
    # Align raw diarization data with discretized chunks
    selected_raw_diar = []
    assigments_of_diar_data_raw = []
    for raw_diar_chunk, dicratized_chunk in zip(diar_data_raw, np_dicratized):
        print(raw_diar_chunk.shape)
        print(dicratized_chunk.shape)
        
        # Find the best permutation
        assigments = find_best_assigment(raw_diar_chunk, dicratized_chunk)
        print("Best assigments:",assigments)

        assigments_of_diar_data_raw.append(assigments)
        
        # Handle the permutation by duplicating some raw segments to match the number of discratized segments
        assigned = raw_diar_chunk[assigments]


        
        print(assigned.shape)
        
        selected_raw_diar.append(assigned)

    selected_raw_diar = np.array(selected_raw_diar)
    print(selected_raw_diar.shape)

    # Transpose and reshape the permutated data for plotting
    selected_raw_diar = selected_raw_diar.transpose(1, 0, 2).reshape(np_dicratized.shape[1], -1).T

    # # Apply threshold to get binary values
    # threshold = 0.5
    # selected_raw_diar = selected_raw_diar > threshold

    # Copy np_dicratized again for plotting
    np_dicratized = discratized_segments.data.copy()

    # Plot them in pairs
    fig = go.Figure()


    for i in range(np_dicratized.shape[1]):
        fig.add_trace(go.Scatter(x=np.arange(np_dicratized.shape[0]) / 100 / 60, y=np_dicratized[:,i] * 0.5 + i, mode='lines', name=f'Mic max {i}'))

    for i in range(selected_raw_diar.shape[1]):
        fig.add_trace(go.Scatter(x=np.arange(selected_raw_diar.shape[0]) / 100 / 60, y=selected_raw_diar[:,i] * 0.5 + i, mode='lines', name=f'Mic raw {i}'))




    fig.update_layout(
        title="Permutated Raw Diarization Data and Discretized Chunks",
        xaxis_title="Time (minutes)",
        yaxis_title="Speakers",
        showlegend=True
    )


    fig.show(renderer="browser")








    #(4801516, 8)

    #switch dims
    original_audio = audio


    data = final_output

    #remove noise channel in dat
    # data = data[:,:,:4,:]

    #keep only mics
    # mics_to_keep = [0, 1, 3, 5]

    data = data[mics_to_keep]
    original_audio = original_audio[mics_to_keep]

    print(data.shape)


    assigments_of_diar_data_raw_np = np.array(assigments_of_diar_data_raw)

    data_assigned = np.zeros((data.shape[0], data.shape[1], assigments_of_diar_data_raw_np.shape[1], data.shape[3]))

    #-> (4, 30, 4, 160000) #x microphones, 30 chunks, x sources, 160000 samples


    
    for mic in range(data.shape[0]):
        for chunk in range(data.shape[1]):
            # print(f"mic {mic}, chunk {chunk}")

            #do reshufling on the sources assigments_of_diar_data_raw_np[i] =  [2, 1, 0, 3]

            data_assigned[mic, chunk] = data[mic, chunk, assigments_of_diar_data_raw_np[chunk], :]

    print(data_assigned.shape)

    data_assigned = einops.rearrange(data_assigned, 'a b c d -> a c (b d)') # mic, assigned_best_source, samples

    print(data_assigned.shape)

    print(assigments_of_diar_data_raw)


    #itter over only segments of speaker 0

 

    result = np.zeros_like(data_assigned)

    # obj = Audio(sample_rate=16000, mono='downmix')


    
    for segment,_,label in merged_annotation_representative_speakers.itertracks(yield_label=True):
        #get index if label
        # print(merged_annotation_max_speakers.labels(),label)

        label_index = merged_annotation_representative_speakers.labels().index(label)
        # print(label_index) #label index is mic and speaker index because the order is the same

        #get seperated data
        seperated_audio = data_assigned[label_index,label_index,:]

    
        # audio,sr = obj.crop({"waveform": torch.from_numpy(seperated_audio).unsqueeze(0), "sample_rate": 16000}, segment,mode="pad") #out of bouds pad with zeros
        # print(audio.shape)

    


        start = segment.start
        end = segment.end

        if start>end:
            end,start = start,end

        #multiply by sr
        start = start * sr
        end = end * sr

        start = int(start)
        end = int(end)

        # print(f"Start: {start}, End: {end}",audio.shape,"diff ",end-start)

        #check if both start and end are over signal
        if start > seperated_audio.shape[0] and end > seperated_audio.shape[0]:
            continue


        #check if end is over signal
        if end > seperated_audio.shape[0]:
            end = seperated_audio.shape[0]

        #check if start and end are the same
        if start == end:
            continue

        

        

        audio = seperated_audio[start:end]
        #check if the audio is really quiet
        max_amplitude = np.max(np.abs(audio))

        #check if close to zero using np all
        if np.allclose(audio,0):
            print("Audio is zero",max_amplitude)
            #use original non seperated audio

            result[label_index,label_index,start:end] = original_audio[label_index,start:end]

        else:


        #set result to seperated audio
            result[label_index,label_index,start:end] = audio



    print(result.shape)

    kept = np.zeros((result.shape[0],result.shape[2]))
    for i in range(result.shape[0]):
    
        kept[i] = result[i,i,:]

    print(kept.shape)


    #save the kept audio
    # sf.write("./outputttt3_seperated_audio.wav",kept.T,sr)

    sf.write(dst_filename.with_name(f"{dst_filename.stem}{dst_filename.suffix}"), kept.T, sr, "PCM_24")




        







if __name__ == "__main__":
    main(add_common_args, initialize, separate)
