# About the Project 

TODO

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

This is a list of the things you need to get started developing in this project:

* [uv](https://docs.astral.sh/uv/getting-started/features/) - handles package and Python versioning, ensuring a consistent experiment environment across machines and operating systems.
  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

* [Git Large File Storage (LFS)](https://git-lfs.com/) - allows us to store model artifact files on git hub. From root run:
    ``` sh
    sudo apt-get install git-lfs
    git lfs install
    git config --global http.timeout 600  

    ```
### Installation

1. Clone the repository (via SSH)
    ```sh
    git clone git@github.com:nedekriek/MoL-ARC-AGI.git
    ```

2. Initialise git submodule 
    ```sh 
    git submodule update --init
    ```  

3. Initialise virtual env
    ``` sh
    uv sync
    ```
### Development 

1. Make sure all Prerequisites are installed.
2. Run scripts with
    ``` sh
    uv run <PATH-TO-PYTHON-FILE>
    ``` 
    Examples:

    1. Run inference with
        ``` sh
        uv run src/submission.py
        ```
    2. Run training with
        ``` sh
        uv run src/submission.py
        ```
#### Running scripts with uv

Please refer to the this [tutorial](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies) for instructions on how to run (and create) scripts with uv. 

#### Adding new packages with uv

To add and remove a python packages to the project use `uv add <package_name>` and `uv remove <package_name>`.

For instance to add `numpy` to the dependency graph run:
```sh
uv add numpy
``` 
To remove `numpy` from the dependency graph run:
```sh
uv remove numpy
```

#### Running JupyterLab with uv 

uv can also be used to run Jupyter notebooks, the same way as we run Python scripts:

```zsh
# If Jupyter Lab isn't already installed, add it as a dependency
uv add jupyterlab

# Start a JupyterLab server
uv run jupyter lab
```

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

TODO