# Prismatic-MISER

[Prismatic](https://github.com/prism-em/prismatic) is a (scanning) transmission electron microscopy simulation software using the PRISM algorithm and the multislice method.  Here, I have added a new mode to Prismatic for performing multislice simulations with very large numbers (~10^4) of frozen phonon configurations.  To average more efficiently over the space of all possible frozen phonon configurations, I make use of the MISER algorithm, a Monte Carlo method that is useful for integrals over large numbers of dimensions.

## FAQ: Why would we ever need this
This is admittedly a very niche thing, but I wanted to see if the extremely weak magnetic signal from antiferromagnetic materials would get drowned out by thermal diffuse scattering in typical atomic-resolution ADF-STEM or 4D-STEM experiments.  (Turns out that it is still present in the simulations, but it *is* so weak and entangled with the atomic signal that it probably could not be detected in real life.)