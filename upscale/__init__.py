"""4x upscalers with the same push()/flush() interface, chosen with --model: vsr (the SR
model + RIFE, see vsr/README.md) and bicubic (plain bicubic interpolation, a fast baseline).

Import the submodules directly; upscale.vsr resolves its symbols lazily, so importing it
does not load torch and diffusers.
"""
