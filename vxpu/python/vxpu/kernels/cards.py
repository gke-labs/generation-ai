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

"""Card memory bandwidth table and lookup matching the Go ledger."""

# pyright: strict

BANDWIDTH_GBS: dict[str, float] = {
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": 1460.0,
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": 1460.0,
    "NVIDIA A100-SXM4-80GB": 1800.0,
    "NVIDIA A100-SXM4-40GB": 1400.0,
    "NVIDIA L4": 260.0,
}


def bandwidth_for(card: str) -> float | None:
    """Returns the card's memory bandwidth in GB/s, or None if unknown.

    Strips 'NVIDIA ' from both the table key and the card name and compares equal.
    """
    card_name = card.removeprefix("NVIDIA ")
    for name, bw in BANDWIDTH_GBS.items():
        if name.removeprefix("NVIDIA ") == card_name:
            return bw
    return None
