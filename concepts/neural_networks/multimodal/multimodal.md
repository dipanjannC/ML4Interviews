# Vision Encoder

## Contrastive learning

Contrastive learning is a self-supervised learning technique where the model learns by comparing similar pairs (positives) and dissimilar pairs (negatives).

Instead of predicting labels, the model learns a representation space where:
- Similar data points are close together.
- Dissimilar data points are far apart.

Example -

Let’s say we are training on images of cats and dogs:
	1.	Take an image of a cat 🐱.
	2.	Apply augmentation (crop, color change, rotation) → another cat image (positive pair).
	3.	Take an image of a dog 🐶 → negative pair.

The model should learn that both cat images are close in vector space, while the dog image is far.