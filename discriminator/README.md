## Starter code for robustqa track
- Download datasets from [here](https://drive.google.com/file/d/1Fv2d30hY-2niU7t61ktnMsi_HUXS6-Qx/view?usp=sharing)

- Setup environment with `conda env create -f environment.yml`
- Train a baseline MTL system with `python train.py --do-train --eval-every 2000 --run-name baseline`
- Evaluate the system on test set with `python train.py --do-eval --sub-file mtl_submission.csv --save-dir save/baseline-01`
- Upload the csv file in `save/baseline-01` to the test leaderboard. For the validation leaderboard, run `python train.py --do-eval --sub-file mtl_submission_val.csv --save-dir save/baseline-01 --eval-dir datasets/oodomain_val`


- Run code test training  with `python train.py --do-train --eval-every 40 --run-name test --train-dir datasets/oodomain_train --val-dir datasets/oodomain_val --train-datasets race,relation_extraction,duorc --adv --enable-length-bp-penalty true --recompute-features`
- Record training commandlines:
- `python train.py --do-train --eval-every 5000 --run-name all-data-full-adv-length-loss-1e-1 --adv --full-adv true --recompute-features --outdomain-data-repeat 3 &&  python train.py --do-train --eval-every 5000 --run-name all-data-full-adv-1e-1 --adv --full-adv true --enable-length-loss false --recompute-features --outdomain-data-repeat 10`
