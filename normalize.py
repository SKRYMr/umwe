import argparse
import os
from itertools import chain
from src.logger import create_logger
from src.utils import load_fasttext_model_for_export, normalize_embeddings

parser = argparse.ArgumentParser(description="FastText conversion")
parser.add_argument("--exp_path", default="", type=str, help="Path to the experiment to normalize")
parser.add_argument("--tgt_lang", type=str, default="en", help="Target language")
parser.add_argument("--src_langs", type=str, nargs="+", default=[], help="Source languages used for training, in the same order as src_embs!")
parser.add_argument("--emb_dim", type=int, default=300, help="Dimension of embeddings")

params = parser.parse_args()

if params.exp_path == "":
    print("No path specified!")
    exit()

norm_path = os.path.join(params.exp_path, "normalized")
if not os.path.exists(norm_path):
    os.mkdir(norm_path)

if not os.path.exists(params.exp_path):
    print("Path does not exist!")
    exit()

log_name = "normalization.log"
logger = create_logger(os.path.join(norm_path, log_name))
logger.info('============ Initialized logger ============')
logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
logger.info('The result will be stored in %s' % norm_path)

for lang in params.src_langs + [params.tgt_lang]:
    model_path = os.path.join(params.exp_path, f"vectors-{lang}.bin")
    _, matrix, model = load_fasttext_model_for_export(params, lang, model_path)
    normalize_embeddings(matrix, "renorm")
    path = os.path.join(norm_path, f"vectors-{lang}.bin")
    logger.info(f"Writing embeddings to {path}...")
    model.set_matrices(matrix.numpy(), model.get_output_matrix())
    model.save_model(path)
    del matrix
    del model

logger.info("DONE!")