# Triple Adversarial Learning for Influence based Poisoning Attack in Recommender Systems
This project is for the paper: [Triple Adversarial Learning for Influence based Poisoning Attack in Recommender Systems](https://dl.acm.org/doi/abs/10.1145/3447548.3467335), Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021: 1830-1840.

The code was developed on Python 3.6 and tensorflow 1.14.0.

## Usage

### run generate_fake.py
```
usage: python main.py [--dataset DATA_NAME] [--gpu GPU_ID]
[--epochs EPOCHS] [--data_size DATA_SIZE] [--target_index TARGET_ITEMS]

optional arguments:
  --dataset DATA_NAME
                        Supported: filmtrust, ml-100k, ml-1m.
  --gpu GPU_ID
                        GPU ID, default is 0.
  --epochs EPOCHS
                        Training epochs.
  --data_size DATA_SIZE
                        The data available to the attacker.
  --target_index TARGET_ITEMS
                        The index of predefined target item list: 0, 1 for ml-100k, 2,3 for ml-1m, 4,5 for filmtrust, 6,7 for yelp.
```

### Example.
```bash
python generate_fake.py --dataset ml-100k --gpu 0 --target_index 0
```