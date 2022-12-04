FROM jetson-voice:r35.1.0

# ===========================================================================================
# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*


# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && sudo apt update

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO noetic

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-core \
    && rm -rf /var/lib/apt/lists/*

# ===========================================================================================
RUN apt update && apt install -y curl && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt update && \
    apt install -y ros-noetic-ros-base
# ===========================================================================================

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install rospkg

RUN python3 -m pip install pyctcdecode && python3 -m pip install https://github.com/kpu/kenlm/archive/master.zip

ENV TZ=America/Sao_Paulo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && apt update && apt install -y tzdata

RUN apt-get install --no-install-recommends -y ffmpeg bash curl git && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

RUN pip install PyYAML==6.0 --ignore-installed 

RUN python -c "import torch; \
               torch.__version__ = torch.__version__.split('a')[0]; \
               torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=False, trust_repo=True)"

ENV TRANSFORMERS_CACHE="/app/.cache/huggingface"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install git+https://github.com/pytorch/fairseq@05255f96410e5b1eaf3bf59b767d5b4b7e2c3a35 

# ENV model=jonatasgrosman/wav2vec2-xls-r-1b-english
# RUN python -c "from os import getenv; from transformers import AutoTokenizer, AutoFeatureExtractor, Wav2Vec2ForCTC; AutoFeatureExtractor.from_pretrained(getenv('model')); tokenizer = AutoTokenizer.from_pretrained(getenv('model')); model = Wav2Vec2ForCTC.from_pretrained(getenv('model'))"

RUN pip install numpy --upgrade && \
    pip install pyloudnorm ffmpeg-python

RUN apt install ffmpeg -y && \
    git clone https://github.com/openai/whisper && \
    cd whisper && pip install . --no-deps && \
    python3 -m pip install more-itertools transformers==4.19.0 && \
    python -c "import whisper; whisper.load_model('base'); whisper.load_model('small'); whisper.load_model('tiny')"

# RUN apt -y update && apt -y upgrade && \
#     apt -y install libsndfile1
RUN python3 -m pip install git+https://github.com/scikit-learn/scikit-learn.git scipy librosa unidecode inflect && \
    python3 -c "import torch; torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16'); \
                torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')"
# ===========================================================================================

# various late additions
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		iproute2 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean


RUN python3 -m pip install netifaces num2words spacy==3.4.1 && python3 -m spacy download en_core_web_sm

RUN apt update && apt install -y python3-catkin-tools

# For wakeword
RUN apt install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt install -y python3.6
################################################################
## project install
################################################################

ARG WORKSPACE=/jetson_voice
COPY jetson-voice/jetson_voice ${WORKSPACE}
ENV PYTHONPATH="${WORKSPACE}:${PYTHONPATH}"

ARG WORKSPACE=/voice_ws
ARG pkg=/src/voice
RUN mkdir -p /${WORKSPACE}/${pkg}
WORKDIR ${WORKSPACE}

COPY ./src ${WORKSPACE}/${pkg}/src
COPY ./launch ${WORKSPACE}/${pkg}/launch
COPY ./msg ${WORKSPACE}/${pkg}/msg
COPY ./srv ${WORKSPACE}/${pkg}/srv

# ===========================================================================================
# setup entrypoint
COPY ./ros_entrypoint.sh /

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]

RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /root/.bashrc 

COPY ./CMakeLists.txt ${WORKSPACE}/${pkg}/
COPY ./package.xml ${WORKSPACE}/${pkg}/
RUN /bin/bash -c '. /opt/ros/$ROS_DISTRO/setup.bash; cd ${WORKSPACE}; catkin build'