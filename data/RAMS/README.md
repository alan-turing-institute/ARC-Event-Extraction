# Roles Across Multiple Arguments (RAMS)

The RAMS data set is available here: https://nlp.jhu.edu/rams/


## Set up of the folder

The folder should be set up as follows:

```
.
└── data/
    └── RAMS/
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
        ├── schema/
        │   ├── pattern.json
        ├── splits/
        │   ├── split1/
        │   │   ├── dev.txt
        │   │   ├── test.txt
        │   │   └── train.txt
        │   └── ...
        └── README.md
```

The data downloaded from RAMS should be saved into the `data/RAMS/data` folder.

The data in the `schema` folder, have been saved manually from [this file](https://github.com/ej0cl6/TextEE/blob/main/TextEE/models/BartGen/pattern.py) and is in the form:

```
{
    "artifactexistence.damagedestroy.damage": {
        "template": "<arg1> damaged <arg2> using <arg3> instrument in <arg4> place",
        "roles": ['damager', 'artifact', 'instrument', 'place']
    },
    ...
}
```

The splits we have used were downloaded from TextEE and were saved into the `data/RAMS/splits` folder, these are found in this repo <https://github.com/ej0cl6/TextEE>
