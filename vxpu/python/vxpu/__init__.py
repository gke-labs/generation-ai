# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vXPU: run PyTorch models on remote accelerators via portable artifacts.

Export locally without weights (meta device), ship a small artifact,
execute wherever the accelerators are.
"""

__version__ = "0.1.0"

__all__ = ["AutoModelForCausalLM", "VxpuModelForCausalLM", "Client"]


def __getattr__(name):
    # Lazy so that `import vxpu` (and the torch-free client) never pulls
    # in torch unless the modeling facade is actually used.
    if name in ("AutoModelForCausalLM", "VxpuModelForCausalLM"):
        from . import modeling
        return getattr(modeling, name)
    if name == "Client":
        from .client import Client
        return Client
    raise AttributeError(f"module 'vxpu' has no attribute {name!r}")
