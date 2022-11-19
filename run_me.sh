mkdir src launch srv msg
rsync -r --existing ../jetson-voice-torch/ros/src/* src/
# rsync -r --existing ../jetson-voice-torch/ros/launch/* launch/
# rsync -r --existing ../jetson-voice-torch/ros/launch/* launch/
rsync -r --existing ../jetson-voice-torch/ros/srv/* srv/
rsync -r --existing ../jetson-voice-torch/ros/msg/* msg/
# rsync -r --existing ../wake_word/src/* src/
# rsync -r --existing ../wake_word/launch/* launch/