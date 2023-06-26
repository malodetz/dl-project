# Code comment generation

To get results run:

0. Load directories `data` and `models` into the `dl-project`
1. `pip install -r requirements.txt`
2. `python -m code2seq.comment_code2seq_wrapper -p models/comment_code2seq.ckpt -c configs/comment-code2seq-transformer.yaml -o output.txt predict`