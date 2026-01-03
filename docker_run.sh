docker run -it \
    --gpus all \
    --ipc=host \
    --name resnet_chlee \
    -v ./:/workspace \
    resnet \
    /bin/bash