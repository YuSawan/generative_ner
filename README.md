# generative_ner
A library for Generative named entity recognition


## Usage

### Instllation
```
git clone git@github.com:YuSawan/generative_ner.git
cd generative_ner

# if venv
python -m venv .venv
source .venv/bin/activate
pip install .

# if uv
uv sync
```

### Dataset preparation
#### Dataset
```
{
  "id": "doc-001",
  "examples": [
    {
      "id": "doc-001-P1",
      "text": "She graduated from NAIST.",
      "entities": [
        {
          "start": 19,
          "end": 24,
          "label": "ORG"
        }
      ],
    }
  ]
}
```


### GPT series
#### OpenAI API
```
python src/cli/run_gpt.py \
    -c configs/config_gpt.yaml
    -o ./output/
    -m generate # {generate, debug, estimate}
```
#### OpenAI BatchAPI
```
# Submit Job to BatchAPI
python src/cli/run_gpt.py \
    -c configs/config_gpt.yaml
    -o ./output/
    -m batch

# Retrieve Job from BatchAPI
python src/cli/run_batchapi.py \
    -c configs/config_gpt.yaml
    -o ./output/
```

### Open LLMs
#### Zero-shot
```
python src/cli/run.py \
    --do_eval \
    --do_predict \
    --config_file configs/config.yaml
    --output_dir ./output/
```

#### Finetuning
```
python src/cli/run.py \
    --do_train \
    --do_eval \
    --do_predict \
    --config_file configs/config.yaml
    --output_dir ./output/
```
