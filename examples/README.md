To run the code for the Omniglot tasks, the steps are:

- Add this repo to the python path as `/path/to/imaml_dev`
- Download the omniglot dataset and place it somewhere (call the path to `omniglot_py` as $DATA_DIR, note that it has to be absolute path and not relative)
- Pre-generate the task definition, where each task is a support and query set with K images each. Run from this directory, i.e.  `imaml_dev/examples`:
```
python generate_task_defs.py --save_dir ./task_defs --N_way 5 --K_shot 1 --num_tasks 5000 --data_dir $DATA_DIR
```
**NOTE:** The above will only generate 5000 tasks to use, which will lead to a lot of overfitting. For replicating results from the paper, you need to use at least 200000 tasks. This will take considerable amount of time (approx. 20 mins), so I'm showing 5000 tasks for example purpose.
- Perform training as:
```
CUDA_VISIBLE_DEVICES=0 python omniglot_implicit_maml.py --save_dir 5_way_1_shot_exp1 --N_way 5 --K_shot 1 --inner_lr 1e-1 --outer_lr 1e-3 --n_steps 16 --meta_steps 500 --num_tasks 5000 --task_mb_size 32 --lam 2.0 --cg_steps 5 --cg_damping 1.0 --load_tasks ./task_defs/Omniglot_5_way_1_shot.pickle --data_dir $DATA_DIR
```
**NOTE:** To replicate the paper results you need to use `meta_steps (at least) 20000` and `num_tasks (at least) 200000`. I'm leaving smaller parameters to help check the code runs properly.
- Finally, to measure the accuracy on test tasks, use
```
CUDA_VISIBLE_DEVICES=0 python measure_accuracy.py --load_agent 5_way_1_shot_exp1/final_model.pickle --N_way 5 --K_shot 1 --num_tasks 600 --n_steps 16 --lam 0.0 --inner_lr 1e-1 --task Omniglot --data_dir $DATA_DIR
```
