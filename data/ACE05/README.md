# ACE 2005 Dataset

This dataset is read by the processing code used in TextEE.
This takes subsections of documents from the ACE 2005 dataset and attaches the relevant annotations to the sentences.
The ACE 2005 dataset is paid for and the format shown here is close to the original format provided.
The difference is that the `outputs`, `splits`, and other files needed to do the pre-processing are included in this folder, along with the output of the pre-processing.

## Set up of the folder

The folder should be set up as follows:

```text
.
└── ACE05/
    ├── data/
    │   └── English
    │       ├── bc
    │       │   ├── FileList(.txt)
    │       │   ├── adj
    |       |   |   ├── CNN_CF_20030303.1900.00.ag.xml
    |       |   |   ├── CNN_CF_20030303.1900.00.apf.xml
    |       |   |   ├── CNN_CF_20030303.1900.00.apf.xml.score
    |       |   |   ├── CNN_CF_20030303.1900.00.apf.sgm
    |       |   |   ├── CNN_CF_20030303.1900.00.apf.tab
    |       |   |   └── ...
    │       │   ├── fp1
    |       |   |   └── ...
    │       │   ├── fp2
    |       |   |   └── ...
    │       │   └── timex2norm
    |       |   |   └── ...
    │       ├── bn
    │       │   └── ...
    │       ├── cts
    │       │   └── ...
    │       ├── nw
    │       │   └── ...
    │       ├── un
    │       │   └── ...
    │       └── wl
    │           └── ...
    ├── docs
    |   ├── README(.txt)
    |   └── file.tbl
    ├── dtd
    |   ├── ace_source_sgml.v1.0.2.dtd
    |   ├── ag-1.1.dtd
    |   └── apf.v5.1.1.dtd
    ├── output/
    │   ├── split1/
    │   │   ├── dev.jsonlines
    │   │   ├── test.jsonlines
    │   │   └── train.jsonlines
    │   └── ...
    ├── splits/
    │   ├── split1/
    │   │   ├── dev.txt
    │   │   ├── test.txt
    │   │   └── train.txt
    │   └── ...
    ├── sent_map.json
    ├── token_map.json
    └── README.md
```

The data downloaded from ACE_2005 should be saved into the `data/ACE05/data` folder.
ACE 2005 also comes with documentation, this should be saved in the `docs` folder.
The DTD files should be saved in the `dtd` folder.

The sentence and token map and the splits we have used were downloaded from TextEE, these are found in this repo <https://github.com/ej0cl6/TextEE>
