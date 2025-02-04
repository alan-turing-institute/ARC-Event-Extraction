# WikiEvents Dataset

The WikiEvents data set is available from an s3 bucket.
Information about WikiEvents can be found here <https://github.com/raspberryice/gen-arg>
Information from TextEE of how to get the data is described here <https://github.com/ej0cl6/TextEE/tree/main/data/preprocessing/wikievents>.


## Set up of the folder

The folder should be set up as follows:

```
.
└── data/
    └── WikiEvents/
        ├── data/
        │   ├── dev.jsonlines
        │   ├── test.jsonlines
        │   └── train.jsonlines
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
        ├── seg_map.json
        └── README.md
```

The data downloaded from WikiEvents should be saved into the `data/WikiEvents/data` folder.

The segment map and the splits we have used were downloaded from TextEE and were saved into the `data/WikiEvents/splits` folder, these are found in this repo <https://github.com/ej0cl6/TextEE>
