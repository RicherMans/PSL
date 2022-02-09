# Downloading audioset script
# Requirements are proxychains and ffmpeg
# Also Youtube-dl
# pip install youtube-dl


# Chan be changed by ./0_download_audioset.sh 8
base_dir=${1-"./data"}
njobs=${2:-4}

SAMPLE_RATE=16000
EXTENSION="wav"

balanced_dir=${base_dir}/audio/balanced/
eval_dir=${base_dir}/audio/eval/
csv_dir=${base_dir}/csvs
log_dir=${base_dir}/logs

fetch_clip() {
    # echo "Fetching $1 ($2 to $3)..."
    outname="$1_$2_$3"
    outdir=${4}
    output_path=${outdir}/${outname}.${EXTENSION}

    # Do not redownload already existing file
    if [ -f "${output_path}" ]; then
        return
    fi
    link=$(youtube-dl -g https://youtube.com/watch?v=$1 | awk 'NR==2{print}')

    if [ $? -eq 0 ]; then
        proxychains -q ffmpeg -loglevel quiet  -i "$link" -ar $SAMPLE_RATE -ac 1 \
        -ss "$2" -to "$3" "${output_path}"
    fi
}


function parallel_download() {
    if [[ $# != 2 ]]; then
        echo "[csv_segments] [output_dir]"
        exit
    fi
    csv_segments=${1}
    output_dir=${2}
    echo "Downloading ${csv_segments} Subset using ${njobs} workers"
    grep "^[^#;]" ${csv_segments} | parallel --bar --resume --joblog ${log_dir}/job.log -j $njobs --colsep=, fetch_clip {1} {2} {3} ${output_dir} > /dev/null
}


export SAMPLE_RATE
export EXTENSION
export -f fetch_clip

mkdir -p ${csv_dir} ${balanced_dir} ${eval_dir} ${log_dir}

wget --continue http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv -O ${csv_dir}/balanced_train_segments.csv
wget --continue http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv  -O ${csv_dir}/eval_segments.csv
wget --continue http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv -O ${csv_dir}/class_labels_indices.csv


parallel_download ${csv_dir}/balanced_train_segments.csv ${balanced_dir}

parallel_download ${csv_dir}/eval_segments.csv ${eval_dir}

echo "Finished Downloading data"




