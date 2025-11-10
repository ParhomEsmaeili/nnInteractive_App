docker container run --gpus all --rm -ti \
--volume /home/parhomesmaeili/IS_Codebase_Forks/nnInteractive_Fork:/workspace/IS_Codebase_Forks/nnInteractive \
--volume /home/parhomesmaeili/IS-Validation-Framework:/workspace/IS-Validation-Framework \
--volume /home/parhomesmaeili/env_bashscripts/is_runscripts/nninteractive_runscript.sh:/workspace/runscripts/nninteractive_runscript.sh \
--volume /home/parhomesmaeili/local_docker_vscode-server:/home/parhomesmaeili/.vscode-server \
--cpus 16 \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--ipc host \
--name nninteractivev1_test \
testing:nninteractivev1

