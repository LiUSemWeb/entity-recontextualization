# Contextualizing Entity Representations
Code, data, and supplementary results for the EKAW 2024 submission "Contextualizing Entity Representations for Zero-Shot Relation Extraction with Masked Language Models."


## Requirements
The python version we used is `3.8.16`.
There is a requirements.txt file with the top-level Python libraries needed for the experiments, the remainder should be dependencies of those and be automatically fetched.
Other libraries may be needed to do the plotting/visualization.
There is a data dependency; see the next section.


## Data
You will need the original DocRED files.
The instructions for that can be found here: https://github.com/thunlp/DocRED.
Place `dev.json` in `data/docred` and optionally include `train_annotated.json`.
We have already included an extended version of `rel_info.json` there.
It includes the prompts that were used in the experiments along with other annotations, some of which are unused and may be incorrect placeholders.

You will also need the original BioRED files.
These can be found at https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/.
Extract the contents of `BIORED.zip` into the folder `data/biored`.
Please do not overwrite `rel_info_full.json`.
You will also need to run the provided conversion script to convert the BioRED data to one that mimicks the DocRED format.


## Experiments
To run the experiments, run the following commands from the root directory:
`python EntitySubstituteTest.py <Task> <Set> <Model Name>     <Passes> <Result Folder> <Batch Size> <Start At>`
`python EntitySubstituteTest.py docred dev   bert-base-cased  2        res             1000         0`
`python EntitySubstituteTest.py docred dev   bert-large-cased 2        res             1000         0`
`python EntitySubstituteTest.py docred dev   roberta-large    2        res             1000         0`
`python EntitySubstituteTest.py biored train bert-large-cased 2        res             1000         0`
`python EntitySubstituteTest.py biored train biobert          2        res             1000         0`
`python EntitySubstituteTest.py biored train pubmedbert       2        res             1000         0`

All parameters have defaults, so the most important parameters to set are `Task`, `Set`, and `Model Name`.
There are some predefined aliases that can be used for the `Model Name`, they are shown above.
For example, `biobert` is an alias for `dmis-lab/biobert-large-cased-v1.1`
These last two are optional and can be adjusted depending on if there are memory issues and what kind those are.
`Start At` starts the experiments at a particular document index, useful if you run out of memory and the batch size needs to be lowered.
This process can also be sped up by running the code on multiple machines and pointing `res` to a shared folder.


## Results
This files currently under `res` contain the results from the paper, including those for the various nonlinearities.
Keep in mind that for `Passes > 0` you will need to read the column `Top-50` instead of `None`.


## Docker
Docker is not necessary to run this project, we simply used docker containers to parallelize the experiments across the machines available to us.
`build.sh`, `run_exp.sh`, and the `images/` folder are all there to support that infrastructure.
However, the underlying python scripts can be run on any CUDA-enabled machine.
If you do wish to run this via docker, an example command would be:
`./build.sh && ./run_exp.sh biored train 0 0 6000 bert-large-cased && docker logs --follow semantics`


## TODO:
The following need to be cleaned up and added still:
1. Score script*
2. Domain and range script