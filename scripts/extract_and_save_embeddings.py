import os

from flx.data.dataset import *
from flx.data.embedding_loader import EmbeddingLoader
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size
from flx.data.image_loader import BaseNLoader
from flx.data.transformed_image_loader import TransformedImageLoader

from flx.extractor.fixed_length_extractor import get_DeepPrint_TexMinu, DeepPrintExtractor

from flx.image_processing.binarization import LazilyAllocatedBinarizer


MODEL_DIR: str = os.path.abspath("models")
DATASET_PATH: str = os.path.abspath("/storage/datasets/full/baseN_aligned/images/")
OUT_DIRS = ['./latents/', './references']
OUT_NAMES = ['latents.npy', 'references.npy']


def get_extractor(model_dir):
    extractor: DeepPrintExtractor = get_DeepPrint_TexMinu(num_training_subjects=8000, num_dims=256)
    extractor.load_best_model(model_dir)

    return extractor


def get_embeddings(extractor, dataset_path, loader):
    image_loader = TransformedImageLoader(
            images=loader(dataset_path),
            poses=None,
            transforms=[
                LazilyAllocatedBinarizer(5.0),
                pad_and_resize_to_deepprint_input_size,
            ],
        )

    image_dataset: Dataset = Dataset(image_loader, image_loader.ids)

    texture_embeddings, minutia_embeddings = extractor.extract(image_dataset)

    # We concatenate texture and minutia embedding vectors
    embeddings = EmbeddingLoader.combine(texture_embeddings, minutia_embeddings)

    return embeddings


def save_embeddings(embeddings, out_path, out_name):
    embeddings.save(out_path, out_name)


if __name__ == "__main__":
    extractor = get_extractor(MODEL_DIR)

    for out_dir, i in OUT_DIRS:
        embeddings = get_embeddings(extractor, DATASET_PATH+out_dir, BaseNLoader)
        embeddings.save(out_dir, OUT_NAMES[i])
