# Code comment generation

Run guide:

1. [Load](https://disk.yandex.ru/d/lfsIcC594QU9Uw) directories `data` and `models` into the `dl-project`
2. `pip install -r requirements.txt`
3. Train code2seq (`wandb init` needed or switch to local logger) `python -m code2seq.comment_code2seq_wrapper -c configs/comment-code2seq-transformer.yaml train` 
4. Get code2seq predictions `python -m code2seq.comment_code2seq_wrapper -p models/comment_code2seq.ckpt -c configs/comment-code2seq-transformer.yaml -o results/code2seq.txt predict`
5. Get codet5 predictions `python -m codeT5.generate_predictions`
6. Get metrics `python -m summarize_results`