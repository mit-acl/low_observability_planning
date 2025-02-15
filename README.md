# Real-Time Pose-Aware Low Observability Path Planning


This is the source code for real-time pose-aware path planning.

## Installation

### Steps
```sh
# Clone the repository
$ git clone https://github.com/mit-acl/low_observability_planning
$ cd low_observability_planning

# Create a virtual environment (optional but recommended)
$ python3.8 -m venv planning_env  # We used python3.8, other versions of python might also work.
$ source planning_env/bin/activate  

# Install the package
$ pip install -e .
# Install required dependencies
$ pip install -r requirements.txt
```

## Running examples with the pretrained policy
A checkpoint is provided in the `checkpoint` folder. To see an example of how to use it, see `notebooks/student_policy_investigation.ipynb`, sample data is provided under `data_mcmc` and `data_rrt`. Note that the paths in the yaml files need to be changed accordingly (see the next section).

Note that under the `notebooks` folder there are several other notebooks that investigate different modules.


## Collect Data and Train a Policy
First navigate to `evasion_guidance` folder.

The first step is to collect paths with RRT*, this can be done via the following command:
```sh
$ python scripts/collect_rrt_data.py --rrt_config params/rrt_data_collection.yaml
```
Note that the `output_path` in the yaml file should be changed to where you want to save the collected data.

Then we need to run MCMC to refine the paths. This can be done via:
```sh
$ python scripts/mcmc_refine.py --rrt_config params/rrt_data_collection.yaml --mcmc_config params/mcmc_refine.yaml
```
In `mcmc_refine.yaml`, the `rrt_data_path` should be changed where you saved your rrt data (i.e. `output_path` in `rrt_data_collection.yaml`). And the `output_path` in `mcmc_refine.yaml` should be changed to where you want to save the refined paths.

To train a student policy, simply run the following command:
```sh
$ python scripts/bc_train.py params/bc_train.yaml
```
Note that the paths in `bc_train.yaml` should be changed accordingly.