import torch

# Load the segmenter checkpoint (no need for GPU)
path = "checkpoints\mhub_segmenter2.pth"
segmenter = torch.load(path, map_location="cpu", weights_only=False)

# Check the keys inside the file
print(segmenter.keys())
print("-----------------------")
print(segmenter['sound_types'])
print("-----------------------")
num_clusters = segmenter['codebook'].shape[0]
missing = [i for i in range(num_clusters) if i not in segmenter['sound_types']]
print("Missing cluster IDs:", missing)


