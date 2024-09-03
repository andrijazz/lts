# LTS

Logit Scaling for OOD detection.

![](https://github.com/andrijazz/lts/blob/main/resources/lts.gif)

## Setup

```bash
# create conda env and install dependencies
$ conda env create -f environment.yml
$ conda activate lts
# set environmental variables
$ export DATASETS=<your_path_to_datasets_folder>
$ export MODELS=<your_path_to_checkpoints_folder>
# download datasets and checkpoints
$ bash scripts/download.sh
```
Please download ImageNet dataset manually to `$DATASET` dir by following [this](https://gist.github.com/bonlime/4e0d236cf98cd5b15d977dfa03a63643) instructions.

## Run
```bash
$ python ood_eval.py --config config/resnet50_config.yml --use-gpu --use-tqdm
```

## Usage

```python
import torch
import torch.nn as nn

from lts import lts

class Net(nn.Module):
    ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        
        # plug-in LTS to get better OOD detection. Function calculates OOD scaling factor s
        s = lts(x)
        
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, s

net = Net()
for i, data in enumerate(testloader):
    inputs, labels = data
    logits, s = net(inputs)
    
    # get ood predictions
    ood_prediction = get_score(logits * s[:, None])
```

## References

```bibtex
@article{djurisic2022extremely,
  title={Extremely simple activation shaping for out-of-distribution detection},
  author={Djurisic, Andrija and Bozanic, Nebojsa and Ashok, Arjun and Liu, Rosanne},
  journal={arXiv preprint arXiv:2209.09858},
  year={2022}
}
```
      
## Citations

If you use our codebase, please cite our work:
