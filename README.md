# How to run our DFINE model
- Install the required dependencies
`pip install -r requirements.txt`

- Make it executable
`chmod +x run_eval.sh`

- Run evaluation on 3 seeds (on test dataset)
`./run_eval.sh`

- If you want to run evaluation on just 1 seed (on test dataset)
`python3 train.py -c configs/dfine/custom/dfine_hgnetv2_s_obj2coco_test.yml --test-only -r output/dfine_hgnetv2_s_obj2coco_custom/best_stg2.pth`