"""4x upscalers with the same push()/flush() interface, chosen with --model: vsr (the SR
model + RIFE, see vsr/README.md) and bicubic (plain bicubic interpolation, a fast baseline).

Import the submodules directly; upscale.vsr resolves its symbols lazily, so importing it
does not load torch and diffusers.
"""

# The output scale of every upscaler (bicubic, the SR model, foveated): the one place it is set.
# The SR model's own factor is checked against it when it loads (stream.runner).
SCALE = 4
