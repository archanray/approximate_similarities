# Matrix approximation algorithms

This repository contains code and models for symmetric similarity matrix approximations.

1. `run_glue` can generate the required similarity matrix. This code is based on

```
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```

2. `direct_nystrom_with_and_without_eig_corr.py` contains all versions of approximation algorithms. You choose and pick whichever to use.

3. `GLUE_BERT_STSB.py` can help you compute downstream task performance measures.

4. `wme_appx.py` can approximate any `n x n` matrix. By putting it in context of the WMD matrix. 

Each files can be run using python `XXXX.py`