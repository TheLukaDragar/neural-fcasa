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

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ArtifactHook
from pyannote.core import Segment, Annotation

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
from pyannote.audio.pipelines.utils.hook import ArtifactHook, ProgressHook
from typing import Any, Mapping, Optional, Text
import plotly.graph_objects as go

from pyannote.core import Segment, Annotation
from pyannote.audio.core.io import Audio

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

    # #pickle the results
    # with open(dst_filename.with_name(f"{dst_filename.stem}_final_output.pkl"), "wb") as f:
    #     pkl.dump(final_output, f)

    # with open(dst_filename.with_name(f"{dst_filename.stem}_diar_all.pkl"), "wb") as f:
    #     pkl.dump(diar_all, f)

    # with open(dst_filename.with_name(f"{dst_filename.stem}_diar_all_raw.pkl"), "wb") as f:
    #     pkl.dump(diar_all_raw, f)

    # with open(dst_filename.with_name(f"{dst_filename.stem}_z_all.pkl"), "wb") as f:
    #     pkl.dump(z_all, f)

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

    pipeline.to(torch.device("mps"))

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

        full_embedings = audio_in_memory["artifact"][
            "embeddings"
        ]  # (num_chunks, local_num_speakers, dimension)
        hard_clusters = audio_in_memory["artifact"][
            "hard_clusters"
        ]  # (num_chunks, local_num_speakers)

        all_full_embedings.append(full_embedings)
        all_hard_clusters.append(hard_clusters)
        all_diars.append(diarization)
        all_centroids.append(embedings)

    all_max_amplitude = []

    for i in range(audio.shape[0]):

        sr = 16000

        current_diar = all_diars[i]

        current_mic = audio[i]

        # current_mic = einops.rearrange(current_mic, "a b c -> b (a c)")

        # current_mic = einops.rearrange(current_mic, "a b -> (a b)")

        current_speaker_max_amplitude = {}

        for turn, _, speaker in current_diar.itertracks(yield_label=True):

            start = int(turn.start * sr)

            end = int(turn.end * sr)

            segment = current_mic[start:end]

            segment = np.array(segment)

            max_amplitude = np.max(np.abs(segment))

            if speaker not in current_speaker_max_amplitude:

                current_speaker_max_amplitude[speaker] = 0

            current_speaker_max_amplitude[speaker] = max(
                max_amplitude, current_speaker_max_amplitude[speaker]
            )

        current_speaker_max_amplitude = dict(
            sorted(
                current_speaker_max_amplitude.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        all_max_amplitude.append(current_speaker_max_amplitude)

    print(all_max_amplitude)

    # Assuming all_diars and all_max_amplitude are already defined
    # Load diarization data
   

    audio_duration = int(audio.shape[1] / sr)

    # Process diarization data
    num_chunks = final_output.shape[1]

    merged_annotation_max_speakers = Annotation()

    # mics_to_keep = [0, 1, 3, 5]

    for i in range(len(all_diars)):
        # x = all_diars[i].discretize(support=Segment(0, 300))

        # if i not in mics_to_keep:
        #     continue

        maxspeaker = list(all_max_amplitude[i].keys())[0]
        # maxspeaker_index = list(x.labels).index(maxspeaker)

        # x.data = x.data[:, maxspeaker_index]

        filtered_annotation = all_diars[i].copy().subset([maxspeaker])
        # Update the merged annotation with the filtered annotation
        filtered_annotation = filtered_annotation.rename_labels(
            mapping={maxspeaker: f"maxSPEAKER_mic_0{i}"}
        )

        merged_annotation_max_speakers.update(filtered_annotation)

    overlaps = merged_annotation_max_speakers.get_overlap()

    # loop throught the merged annotation and add the segments to the plot each speaker discratized in different y

   


    # Load the diarization data
    diar_data_raw = diar_all_raw.copy()

    #remove the diarization of noise since its always 1 its in postition5

    diar_data_raw = diar_data_raw[:, :5, :]
    print(diar_data_raw.shape)



    discratized_segments = merged_annotation_max_speakers.discretize(support=Segment(0, audio_duration))

    # Assuming discratized_segments is already defined and loaded
    np_dicratized = discratized_segments.data.copy()

    print(np_dicratized.shape)

    # # Keep only the required microphones
    # mics_to_keep = [0, 1, 3, 5]
    # np_dicratized = np_dicratized[:, mics_to_keep]

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

    # data = data[mics_to_keep]
    # original_audio = original_audio[mics_to_keep]

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


    for segment,_,label in merged_annotation_max_speakers.itertracks(yield_label=True):
        #get index if label
        # print(merged_annotation_max_speakers.labels(),label)

        label_index = merged_annotation_max_speakers.labels().index(label)
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

        #check if end is over signal
        if end > seperated_audio.shape[0]:
            end = seperated_audio.shape[0]

        

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

    sf.write(dst_filename.with_name(f"{dst_filename.stem}_mic_{dst_filename.suffix}"), kept.T, sr, "PCM_24")




        







if __name__ == "__main__":
    main(add_common_args, initialize, separate)
