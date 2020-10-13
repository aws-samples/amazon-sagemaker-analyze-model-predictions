import smdebug.pytorch as smd
import torch

class CustomHook(smd.Hook):

    def image_gradients(self, image):
        """Register input image for backward pass, to get image gradients"""
        image.register_hook(self.backward_hook("image"))

    def forward_hook(self, module, inputs, outputs):
        module_name = self.module_maps[module]   
        self._write_inputs(module_name, inputs)

        outputs.register_hook(self.backward_hook(module_name + "_output"))

        #record running mean and var of BatchNorm layers
        if isinstance(module, torch.nn.BatchNorm2d):
            self._write_outputs(module_name + ".running_mean", module.running_mean)
            self._write_outputs(module_name + ".running_var", module.running_var)

        self._write_outputs(module_name, outputs)
        self.last_saved_step = self.step
