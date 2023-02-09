# Adversarial_Topic_Model
Experimentation of ATM: Adversarial-neural Topic Model

## Motivation
Generative Adversarial Networks are successful in producing high quality images. A known issue of GAN’s is its inability to generate data with diverse outputs, a phenomenon known as Mode Collapse in GANs. This repository contains the experiments to mitigate this issue using the broad idea of multiple generators to learn all the diverse modes and then using a single discriminator learning to effectively distinguish between data of various different modes. Changes to generator loss function are made to incorporate efficient training of all the generators.
In order to perform this experiment, we take the problem of document modeling and use the data from 20 newsgroups dataset. We aim for the GAN setup to learn the various kinds of topic distributions to generate documents.

## Scripts
The scripts for this experiment are present in the `scripts` folder. They implement the various experimental settings of multiple generators and single discriminator.
