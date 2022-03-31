import argparse
import os
import torch

from src.logger import create_logger
from src.utils import apply_mapping_to_fasttext, load_fasttext_model_for_export, export_fasttext_embeddings, normalize_embeddings
from torch import nn

parser = argparse.ArgumentParser(description="FastText conversion")
parser.add_argument("--exp_path", default="", type=str, help="Path to the experiment to remap")
parser.add_argument("--tgt_lang", type=str, default="en", help="Target language")
parser.add_argument("--tgt_emb", type=str, default="", help="Path to target embeddings")
parser.add_argument("--src_embs", type=str, nargs="+", default=[], help="Path to the source embeddings used for umwe training")
parser.add_argument("--src_langs", type=str, nargs="+", default=[], help="Source languages used for training, in the same order as src_embs!")
parser.add_argument("--max_vocab", type=int, default=-1, help="Maximum number of words to load from embeddings vocabulary")
parser.add_argument("--emb_dim", type=int, default=300, help="Dimension of embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings")

params = parser.parse_args()

assert len(params.src_embs) == len(params.src_langs), "Source languages and paths do not match!"

src_paths = {lang: path for lang, path in zip(params.src_langs, params.src_embs)}
if params.exp_path == "":
    print("No path specified!")
    exit()

if not os.path.exists(params.exp_path):
    print("Path does not exist!")
    exit()

log_name = "conversion.log"
logger = create_logger(os.path.join(params.exp_path, log_name))
logger.info('============ Initialized logger ============')
logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
logger.info('The result will be stored in %s' % params.exp_path)

tgt_lang = params.tgt_lang
for lang in params.src_langs + [tgt_lang]:

    if os.path.isfile(os.path.join(params.exp_path, f"vectors-{lang}.bin")):
            continue
        
    if lang != params.tgt_lang:

        mapping_path = os.path.join(params.exp_path,
                            f"best_mapping_{lang}2{tgt_lang}.t7")
        
        logger.info(f"Reloading the best {lang} to {tgt_lang} mapping from {mapping_path} ...")

        # reload the mapping
        assert os.path.isfile(mapping_path), f"Missing mapping file for lang {lang}!"
        W = torch.from_numpy(torch.load(mapping_path))
        mapping = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
        mapping_weight = mapping.weight
        assert mapping_weight.size() == W.size(), "Mappings do not match embeddings dimension!"
        with torch.no_grad():
            mapping_weight.copy_(W.type_as(mapping_weight))
        del W

        logger.info(f"Mapping loaded.")

        # reload the source embeddings
        emb_path = src_paths[lang]
        logger.info(f"Reloading source embeddings for lang {lang} from {emb_path} ...")
        assert os.path.isfile(emb_path), f"Missing embedding path for lang {lang}!"
        dico, input_matrix, model = load_fasttext_model_for_export(params, lang, emb_path)
        del dico
        logger.info(f"Applying mapping to lang {lang} ...")
        input_matrix = apply_mapping_to_fasttext(mapping, input_matrix)
        logger.info(f"Exporting lang {lang} to {os.path.join(params.exp_path, f'vectors-{lang}.bin')} ...")
        export_fasttext_embeddings(model, input_matrix.numpy(), lang, params)
        logger.info(f"Successfully exported lang {lang}!")
        logger.info("-"*20)


    else:
        
        logger.info(f"Reloading target embeddings for lang {lang} from {params.tgt_emb} ...")
        dico, input_matrix, model = load_fasttext_model_for_export(params, lang, params.tgt_emb)
        del dico
        logger.info(f"Exporting lang {lang} to {os.path.join(params.exp_path, f'vectors-{lang}.bin')} ...")
        export_fasttext_embeddings(model, input_matrix.numpy(), lang, params)
        logger.info(f"Successfully exported lang {lang}!")
        logger.info("-"*20)

logger.info("ALL EMBEDDINGS SUCCESSFULLY EXPORTED!")