"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations


def load_predict_unit(path, *, force_fp32=False):
    """
    Use FP32 forward activations and measured narrow adjoints.
    """
    import torch
    from torch.optim.swa_utils import AveragedModel

    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit

    from ._inference import UMANitroModel

    settings = InferenceSettings(
        tf32=False,
        activation_checkpointing=False,
        merge_mole=True,
        compile=False,
        execution_mode=None,
        base_precision_dtype=torch.float32,
        internal_graph_gen_version=2,
        auto_add_default_untrained_tasks=False,
    )

    class NitroPredictUnit(MLIPPredictUnit):
        def _load_model(self):
            UMANitroModel.validate(
                self.inference_settings,
                self._inference_model_path,
                self._requested_device,
            )
            checkpoint = torch.load(
                self._inference_model_path, map_location="cpu", weights_only=False
            )
            # OrbMol changes Torch's process default dtype. Construct Nitro in
            # its declared precision, as the ordinary FairChem loader does.
            previous = torch.get_default_dtype()
            try:
                torch.set_default_dtype(self.inference_settings.base_precision_dtype)
                self.model = AveragedModel(UMANitroModel.from_checkpoint(checkpoint))
            finally:
                torch.set_default_dtype(previous)
            self.model.eval()

        def _lazy_init(self, data):
            super()._lazy_init(data)
            from ._inference import set_force_precision

            set_force_precision(force_fp32)

        def _process_outputs(self, data, output, undo_refs):
            # The upstream Nitro release permits omitted tasks belonging to
            # other datasets. The older XPU fork's predictor predates that fix.
            result = {}
            for name, task in self.model.module.tasks.items():
                if name not in output:
                    continue
                value = task.normalizer.denorm(output[name][task.property])
                if self.assert_on_nans and not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite Nitro output: {name}")
                if undo_refs and task.element_references is not None:
                    value = task.element_references.undo_refs(data, value)
                result[name] = value
            return result

    return NitroPredictUnit(path, device="xpu", inference_settings=settings)
