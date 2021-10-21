# TPP-Select
### Code structure:
- assets: The dataset, intermediate results and generated figures
- func: Common functions and constants for all modules
- greedy: Greedy algorithm
- model: Linear and nonlinear models used in experiments
- predict: Event prediction for real experiments
- scripts: Executable scripts. Note that all the scripts should be executed in work directory `/TPP-Select/`

### Usage: 
#### For synthetic experiment:
You can generate synthetic experiment in paper through command `pyhton scripts/run_synthetic_visualization.py`. <br /> 
Sample figure is already in `/assets/synthetic_visualization/`

#### For real experiment: 
1. First run greedy algorithm to generate exogenous prediction that will be dumped in `assets/greedy/result/`. <br /> 
Sample result is already there, if you want to run your own greedy, please execute `/scripts/run_real_greedy.py` by <br /> 
`python scripts/run_real_greedy.py --filename [filename] --omega [omega] --v [v] --penalty_time [penalty_time] --penalty_mark [penalty_mark]`, <br /> 
in which the first argument is dataset name and last four arguments are model hyperparameters.
2. Run prediction using `python scripts/run_prediction.py --filename [filename] --base_model [base_model]`.