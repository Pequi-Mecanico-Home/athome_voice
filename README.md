# Athome Voice using torch models

This branch has implementations of pipelines for the human-robot interaction.

To check the development updates, check the voice-ros-torch-dev branches.

## Running

To run the services one can simply run the docker compose using the following command:

```sh
docker-compose up
```

This will automatically download the image and run the service, however, if you want to make modifications, you will need to create the image using the provided Dockerfile.

### Running the services

To run the services, open the terminal of the running container and launch a service, for example: 

```sh
docker exec -it <name of the container> bash
```

Then, inside the container:

```sh
source devel/setup.bash && roslaunch voice tts.launch &
rosservice call voice/tts
```
