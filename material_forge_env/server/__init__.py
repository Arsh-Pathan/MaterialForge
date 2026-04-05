# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Material Forge Env environment server components."""

try:
    from .material_forge_env_environment import MaterialForgeEnvironment
except ImportError:
    try:
        from material_forge_env_environment import MaterialForgeEnvironment
    except ImportError:
        from server.material_forge_env_environment import MaterialForgeEnvironment

__all__ = ["MaterialForgeEnvironment"]
