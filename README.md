## Less is More: Discovering Concise Network Explanations (DCNE)

This code reproduces the results for the paper Less is More: Discovering Concise Network Explanations (DCNE) presented at the First Re-Align Workshop at ICLR 2024.

### Environment
```
conda create -n "DCNE" python=3.10.10
conda activate DCNE
pip install -r requirements.txt
```

### Download checkpoints, annotations, and datasets
```
mkdir checkpoints
TODO add download links

mkdir data
cd data
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1
tar -xzvf CUB_200_2011.tgz
rm CUB_200_2011.tgz
```

### Run minimal example
```
bash ./minimal_example.sh
```
