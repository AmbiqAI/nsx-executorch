# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#
# NS-profile variant of the stock CortexMTester: same harness, but the
# QUANTIZE and RUN_PASSES stages use the cortex_m_ns quantizer and pass
# manager. Everything composes the pinned ExecuTorch tree via public APIs.

from functools import partial
from typing import Optional

from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.backends.cortex_m.test.tester import (  # noqa: F401  (re-exports)
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)
from executorch.backends.test.harness.stages import Quantize, RunPasses, StageType

from nsx_cortex_m.pass_manager_ns import NsCortexMPassManager
from nsx_cortex_m.quantizer_ns import NsCortexMQuantizer


class NsCortexMQuantize(Quantize):
    def __init__(self, calibration_samples=None):
        super().__init__(
            NsCortexMQuantizer(), calibration_samples=calibration_samples
        )


class NsCortexMRunPasses(RunPasses):
    def __init__(self, target_config: Optional[CortexMTargetConfig] = None):
        target_config = target_config or CortexMTargetConfig(cpu=CortexM.M55)
        super().__init__(
            partial(NsCortexMPassManager, target_config=target_config),  # type: ignore[arg-type]
            NsCortexMPassManager.pass_list,  # type: ignore[arg-type]
        )


class NsCortexMTester(CortexMTester):
    def __init__(
        self,
        module,
        example_inputs,
        target_config: Optional[CortexMTargetConfig] = None,
    ):
        super().__init__(module, example_inputs, target_config=target_config)
        self.stage_classes[StageType.QUANTIZE] = NsCortexMQuantize
        self.stage_classes[StageType.RUN_PASSES] = lambda: NsCortexMRunPasses(
            target_config=target_config
        )
