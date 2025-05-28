# Lint as: python3
"""Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition"""

import os
from typing import Any, Iterable, Iterator, Union

import datasets

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{tjong-kim-sang-de-meulder-2003-introduction,
    title = "Introduction to the {C}o{NLL}-2003 Shared Task: Language-Independent Named Entity Recognition",
    author = "Tjong Kim Sang, Erik F.  and
      De Meulder, Fien",
    booktitle = "Proceedings of the Seventh Conference on Natural Language Learning at {HLT}-{NAACL} 2003",
    year = "2003",
    url = "https://www.aclweb.org/anthology/W03-0419",
    pages = "142--147",
}
"""

_DESCRIPTION = """\
The shared task of CoNLL-2003 concerns language-independent named entity recognition. We will concentrate on
four types of named entities: persons, locations, organizations and names of miscellaneous entities that do
not belong to the previous three groups.
The CoNLL-2003 shared task data files contain four columns separated by a single space. Each word has been put on
a separate line and there is an empty line after each sentence. The first item on each line is a word, the second
a part-of-speech (POS) tag, the third a syntactic chunk tag and the fourth the named entity tag. The chunk tags
and the named entity tags have the format I-TYPE which means that the word is inside a phrase of type TYPE. Only
if two phrases of the same type immediately follow each other, the first word of the second phrase will have tag
B-TYPE to show that it starts a new phrase. A word with tag O is not part of a phrase. Note the dataset uses IOB2
tagging scheme, whereas the original dataset uses IOB1.
For more details see https://www.clips.uantwerpen.be/conll2003/ner/ and https://www.aclweb.org/anthology/W03-0419
"""

_URL = "https://data.deepai.org/conll2003.zip"
_TRAINING_FILE = "train.txt"
_DEV_FILE = "valid.txt"
_TEST_FILE = "test.txt"


class Conll2003Config(datasets.BuilderConfig):
    """BuilderConfig for Conll2003"""

    def __init__(self, **kwargs: str) -> None:
        """BuilderConfig forConll2003.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Conll2003Config, self).__init__(**kwargs)


class Conll2003(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset."""

    BUILDER_CONFIGS = [
        Conll2003Config(
            name="conll2003_en",
            version=datasets.Version("1.0.0"),
            description="Conll2003 dataset (English)",
        ),
    ]

    def _info(self) -> None:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string", id=None),
                    "examples": [{
                        "id": datasets.Value("string", id=None),
                        "text": datasets.Value("string", id=None),
                        "entities": [
                            {
                                'start': datasets.Value("int64", id=None),
                                'end': datasets.Value("int64", id=None),
                                'label': datasets.Value("string", id=None),
                                'text': datasets.Value("string", id=None),
                            }
                        ],
                        "word_positions": datasets.Sequence(
                            datasets.Sequence(datasets.Value("int64"))
                        ),
                    }]
                },
            ),
            supervised_keys=None,
            homepage="https://www.aclweb.org/anthology/W03-0419/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> list[Any]:
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract(_URL)
        data_files = {
            "train": os.path.join(downloaded_file, _TRAINING_FILE),
            "dev": os.path.join(downloaded_file, _DEV_FILE),
            "test": os.path.join(downloaded_file, _TEST_FILE),
        }

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": data_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"]}),
        ]

    def _generate_examples(self, filepath: str) -> Iterator[tuple[int, dict[str, Any]]]:
        logger.info("⏳ Generating examples from = %s", filepath)
        basename = os.path.basename(filepath).split('.')[0]
        guid = 0
        for i, sentences in enumerate(read_conll(filepath)):
            document = {"id": f"{basename}-{i}", "examples": []}
            for j, sentence in enumerate(sentences):
                document["examples"].append({
                    "id": f"{document['id']}-{j}",
                    "text": sentence['text'],
                    "entities": sentence['entities'],
                    "word_positions": sentence['word_positions'],
                })
            yield guid, document
            guid += 1


def read_conll(file: Union[str, bytes, os.PathLike]) -> Iterable[list[dict[str, Any]]]:
    sentences: list[dict[str, Any]] = []
    words: list[str] = []
    labels: list[str] = []

    with open(file, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("-DOCSTART"):
                if sentences:
                    yield sentences
                    sentences = []
            elif not line:
                if words:
                    sentences.append(_conll_to_example(words, labels))
                    words = []
                    labels = []
            else:
                cols = line.split(" ")
                words.append(cols[0])
                labels.append(cols[-1])

    if sentences:
        yield sentences


def _conll_to_example(words: list[str], tags: list[str]) -> dict[str, Any]:
    text, positions = _conll_words_to_text(words)
    entities = [
        {"start": positions[start][0], "end": positions[end - 1][1], "label": label, 'text': text[positions[start][0]: positions[end - 1][1]]}
        for start, end, label in _conll_tags_to_spans(tags)
    ]
    return {"text": text, "entities": entities, "word_positions": positions}


def _conll_words_to_text(words: Iterable[str]) -> tuple[str, list[tuple[int, int]]]:
    text = ""
    positions = []
    offset = 0
    for word in words:
        if text:
            text += " "
            offset += 1
        text += word
        n = len(word)
        positions.append((offset, offset + n))
        offset += n
    return text, positions


def _conll_tags_to_spans(tags: Iterable[str]) -> Iterable[tuple[int, int, str]]:
    # NOTE: assume BIO scheme
    start, label = -1, None
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag == "O":
            if start >= 0:
                assert label is not None
                yield (start, i, label)
                start, label = -1, None
        else:
            cur_label = tag[2:]
            if tag.startswith("B"):
                if start >= 0:
                    assert label is not None
                    yield (start, i, label)
                start, label = i, cur_label
            else:
                if cur_label != label:
                    if start >= 0:
                        assert label is not None
                        yield (start, i, label)
                    start, label = i, cur_label


if __name__ == "__main__":
    import json
    import os
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    from pathlib import Path
    parser = ArgumentParser(
        description="Conll2003 Data Preprocessing",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--output', '-o', required=True, metavar='DIR', type=str)
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)

    Conll2003(cache_dir='./tmp').download_and_prepare()
    dataset = Conll2003(cache_dir='./tmp').as_dataset()
    for file in dataset.keys():
        filename = output_path / f'{file}.jsonl'
        with open(filename, 'w') as wf:
            cur_id = None
            for data in dataset[file]:
                wf.write(json.dumps(data)+'\n')
