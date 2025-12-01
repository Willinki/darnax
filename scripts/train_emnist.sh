# adapt as needed

conda run -p "$CONDA_ENV" python ${PROJECT_ROOT}/scripts/train.py \
    model.kwargs.dim_hidden=6000 \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=1.25 \

conda run -p "$CONDA_ENV" python ${PROJECT_ROOT}/scripts/train.py \
    model.kwargs.dim_hidden=5000 \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=1.75 \

conda run -p "$CONDA_ENV" python ${PROJECT_ROOT}/scripts/train.py \
    model.kwargs.dim_hidden=4000 \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=2.0 \

conda run -p "$CONDA_ENV" python ${PROJECT_ROOT}/scripts/train.py \
    model.kwargs.dim_hidden=3000 \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=1.75 \

conda run -p "$CONDA_ENV" python ${PROJECT_ROOT}/scripts/train.py \
    model.kwargs.dim_hidden=2000 \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=2.5 \

conda run -p "$CONDA_ENV" python ${PROJECT_ROOT}/scripts/train.py \
    model.kwargs.dim_hidden=1000 \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=3.5 \

conda run -p "$CONDA_ENV" python ${PROJECT_ROOT}/scripts/train.py \
    model.kwargs.dim_hidden=500 \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=4.25 \

conda run -p "$CONDA_ENV" python ${PROJECT_ROOT}/scripts/train.py \
    model.kwargs.dim_hidden=250 \
    'wandb.tags=[ours,emnist,scalingH]' \
    model.kwargs.strength_back=5.5 \