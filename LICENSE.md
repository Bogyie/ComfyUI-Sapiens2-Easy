# License

This repository contains a ComfyUI adapter for Sapiens2. The adapter code in this
repository is licensed under the MIT License below.

This license applies only to this repository's original wrapper code,
documentation, and packaging files. It does not grant any rights to Meta's
Sapiens2 models, weights, upstream source code, algorithms, documentation, or
other Sapiens2 Materials.

## MIT License for This Adapter

Copyright (c) 2026 bogyie and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Sapiens2 Materials

Sapiens2 is owned and distributed by Meta under the Sapiens2 License:

https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md

If you clone, download, load, use, modify, or redistribute any Sapiens2
Materials, including model weights, upstream Sapiens2 source code, inference
code, training code, fine-tuning code, algorithms, or documentation, you must
comply with the Sapiens2 License.

In particular, the Sapiens2 License includes separate terms for redistribution,
research acknowledgement, privacy and biometric-information laws, trade
controls, termination, audit, warranty, liability, and prohibited uses. Those
terms are not replaced or relaxed by this MIT License.

This repository's `install.py` helper can clone the upstream Sapiens2 repository
into `vendor/sapiens2`, and the Hugging Face nodes can download Sapiens2
checkpoints. Those downloaded or cloned materials remain governed by Meta's
Sapiens2 License and are intentionally excluded from this repository's MIT
license.

## Other Third-Party Software and Models

This project also integrates with ComfyUI, PyTorch, Hugging Face libraries, the
DETR person detector used by the pose pipeline, and other third-party packages.
Those components are governed by their own licenses and terms. You are
responsible for reviewing and complying with those terms when installing,
using, or redistributing them.

