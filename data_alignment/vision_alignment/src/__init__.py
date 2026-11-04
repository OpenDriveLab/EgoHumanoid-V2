"""Human-to-robot visual alignment.

pipeline: CLI and stage orchestration.
segment / masks: SAM detections and generation/compositing masks.
prepare: joint projection followed by image-guided arm placement.
inpaint: ProPainter prior and DiffuEraser background reconstruction.
render: mesh rendering, image-space alignment, and final compositing.
"""
