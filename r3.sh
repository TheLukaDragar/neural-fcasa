#!/bin/bash

source /home/luka/miniconda3/etc/profile.d/conda.sh
conda activate neural

# Get the first argument
filename=$1

# Check if the filename is provided
if [ -z "$filename" ]; then
    echo "Please provide an audio file for dereverberation and separation"
    exit 1
fi

# Define the dereverberated file name
deverb_filename="${filename}_deverb.wav"
separated_filename="${filename}_separated.wav"

# Check if the dereverberated file already exists
if [ -f "$deverb_filename" ]; then
    echo "Dereverberated file already exists: $deverb_filename"
else
    echo "Doing dereverberation"
    python3 -m neural_fcasa.dereverberate "$filename" "$deverb_filename"
    echo "Done dereverberation"
fi

echo "Doing separation"
python3 neural_fcasa/separate_batch_dela3_enhanced_selection.py one hf://b-sigpro/neural-fcasa --dump_diar --out_ch 3 --device cuda --thresh 0.2 "$deverb_filename" "$separated_filename"
echo "Done separation"

#saved to 
echo "Saved to $separated_filename"