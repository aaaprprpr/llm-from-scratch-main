# Vendored Laya runtime

Copied from the sibling `laya` checkout at commit `6d942c9` (version 0.3.22). License: [Apache-2.0](LICENSE). The original package is available at <https://github.com/NandhaKishorM/laya>.

The cleaning tool imports this local package directly. The public `multilingual` checkpoint is stored in `dataset/label/models/laya_multilingual/`. The cleaning tool selects the separately fine-tuned `dataset/label/models/laya_wiki_cleaning_v1/` checkpoint. Both weight directories are ignored by Git because each is about 647 MB. See the [fine-tuning evaluation](../docs/laya-finetuning-evaluation-20260930.md) for the training data, quality limits, and reproduction commands.
