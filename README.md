# NEURAL FCASA based speaker separation

This approach is based on using NEURAL FCASA model that is said to be fast for source separation on 8 microphones.
It separates each mic into 6 sources where 6 is noise. It works on chunks of 10s best and ouputs separated sources for each mic as well as the diarization of the sources. 

The main problem is that the permutation of the sources is not fixed so the output sources are not always in the same order.

This is then solved by ussing pyannotate on each microphone separatly to obtain speaker diarization.
Then for each mic we select the best speaker based on snr and also do clustering to really get the represantative sepakers for each mic, it also works with 2 spekers on same mic.

globaly these speakers are selected and their diarization is then used to find the best coresponding chunk of the separated sources based on rmse beetwen the diarization of separated sources and the diarization of the speakers.

Finally a mask is created where we use separated sources and where we use original audio too.

the final outpu is each speaker in its own channel. Problems arrise where pyannonate fails to detect speakers or where the speakers are not well separated by the neural model.

The algorithm was tested on 5min wav audio clips of sample rate 16000.

```bash
./r7.sh deve_full.wav

```

The output Saved to deve_full.wav_separated.wav