import pyaudio

audio = pyaudio.PyAudio()
info = audio.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i))
# stream_in = audio.open(
#     input_device_index=24,
#     input=True, output=False,
#     format=audio.get_format_from_width(
#         detector.BitsPerSample() / 8),
#     channels=detector.NumChannels(),
#     rate=detector.SampleRate(),
#     frames_per_buffer=2048,
#     stream_callback=audio_callback)