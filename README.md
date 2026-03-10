Build container with:

`podman build --tag mymamba .`

To run the container:

`podman run --rm -it --gpus all --name my_mamba my_mamba`

Make sure to mount a volume in the container to save your work:

`podman run --rm -it --gpus all -v ~/local_path:/app/mount_name:z --name my_mamba my_mamba`
