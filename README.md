Task is use 6 x 1500(1501) sequence feature to predict the start time and end time

There are two approach

1. segementation

if the time interval is between start time and end time, then bit = 1, otherwise = 0

```
python3 run_segementation_task.py --model PatchTST --GPU 0 
```

2. Regression

Get two timepoints directly

```
python3 run_regression_task.py --model PatchTST --GPU 0 
```
