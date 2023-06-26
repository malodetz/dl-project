# Code comment generation

Run guide:

1. Load directories `data` and `models` into the `dl-project`
2. `pip install -r requirements.txt`
3. Train code2seq (`wandb init` needed or switch to local logget)
4. Get code2seq predictions `python -m code2seq.comment_code2seq_wrapper -p models/comment_code2seq.ckpt -c configs/comment-code2seq-transformer.yaml -o results/code2seq.txt predict`
5. Get codet5 predictions `python -m codeT5.generate_predictions`