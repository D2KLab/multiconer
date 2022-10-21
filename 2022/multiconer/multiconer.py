import datasets
import textwrap
import os


_CITATION = """"""
_CITATIONS = {
    "NER": textwrap.dedent(
        (
            """
            """
        )
    )
}

_LANG = {
    "bn": "BN-Bangla",
    "de": "DE-German",
    "en": "EN-English",
    "es": "ES-Spanish",
    "fa": "FA-Farsi",
    "hi": "HI-Hindi",
    "ko": "KO-Korean",
    "mix": "MIX_Code_mixed",
    "multi": "MULTI_Multilingual",
    "nl": "NL-DUTCH",
    "ru": "RU-Russian",
    "tr": "TR-Turkish",
    "zh": "ZH-Chinese"
}

_NAMES = []

_DATA_URLS = {
    "NER": "https://drive.google.com/uc?export=download&id=1KhzP6KvjA848SONZSpP8Szh4cbVWxSl5"
}

for lang in _LANG:
    _NAMES.append(f"NER.{lang}")

_DESCRIPTION = """"""

_DESCRIPTIONS = {
    "NER": textwrap.dedent(
        """\
    MulticoNER task for the SemEval 2022 @ TACL"""
    ),
}

_TEXT_FEATURES = {
    "NER": {"tokens": "", "ner_tags": "", "lang": ""},
}

class MultiCoNERConfig(datasets.BuilderConfig):
    """BuilderConfig for Break"""

    def __init__(self, citation, data_url, text_features, **kwargs):
        """
        Args:
            text_features: `dict[string, string]`, map from the name of the feature
        dict for each text field to the name of the column in the tsv file
            label_column:
            label_classes
            **kwargs: keyword arguments forwarded to super.
        """
        super(MultiCoNERConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.text_features = text_features
        self.citation = citation
        self.data_url = data_url


class MultiCoNER(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.1.0")
    BUILDER_CONFIGS = [
        MultiCoNERConfig(
            name=name,
            description=_DESCRIPTIONS[name.split(".")[0]],
            citation=_CITATIONS[name.split(".")[0]],
            text_features=_TEXT_FEATURES[name.split(".")[0]],
            data_url=_DATA_URLS[name.split(".")[0]],
        )
        for name in _NAMES
    ]

    def _info(self):
        features = {text_feature: datasets.Value("string") for text_feature in self.config.text_features.keys()}
        if self.config.name.startswith("NER"):
            features = datasets.Features(
                {
                    "guid": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-PER",
                                "I-PER",
                                "B-CW",
                                "I-CW",
                                "B-CORP",
                                "I-CORP",
                                "B-GRP",
                                "I-GRP",
                                "B-PROD",
                                "I-PROD",
                                "B-LOC",
                                "I-LOC",
                            ]
                        )
                    ),
                    "langs": datasets.Sequence(datasets.Value("string")),
                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=self.config.description + "\n" + _DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                features
                # These are the features of your dataset like images, labels ...
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            citation=self.config.citation + "\n" + _CITATION,
        )

    def _split_generators(self, dl_manager):
        if self.config.name.startswith("NER"):
            ner_dir = dl_manager.download_and_extract(self.config.data_url)
            lang = self.config.name.split(".")[1]
            lang_folder = os.path.join(ner_dir, _LANG[lang])

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={"filepath": os.path.join(lang_folder, lang + "_dev.conll")},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={"filepath": os.path.join(lang_folder, lang + "_train.conll")},
                ),
            ]

    def _generate_examples(self, filepath):
        lang = filepath.split("/")[-1].split("_")[0]

        with open(filepath, encoding="utf-8") as f:
            tokens = []
            ner_tags = []
            langs = []
            guid_index = 0
            for line in f:
                if line.startswith("# id"):
                    guid_index = line.split("\t")[0].split()[2]
                    continue
                if line == "" or line == "\n":
                    if tokens:
                        yield guid_index, {
                            "guid": guid_index,
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                            "langs": langs,
                        }
                        tokens = []
                        ner_tags = []
                        langs = []
                else:
                    # pan-x data is tab separated
                    splits = line.split()
                    # strip out en: prefix
                    langs.append(lang)
                    tokens.append(splits[0])
                    ner_tags.append(splits[-1].replace("\n", ""))
            if tokens:
                yield guid_index, {
                    "guid": guid_index,
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                    "langs": langs,
                }
