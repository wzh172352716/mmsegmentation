import logging
from mmengine.logging import print_log
from mmengine.runner import IterBasedTrainLoop
from mmengine.registry import LOOPS


class IterBasedTrainLoopFastRestart(IterBasedTrainLoop):

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook('before_train_epoch')
        if self._iter > 0:
            print_log(
                f'Advance dataloader {self._iter} steps to skip data '
                'that has already been trained with fast forward',
                logger='current',
                level=logging.WARNING)
            """self.dataloader_iterator._dataloader.dataset.fast_forward = True
            print(self.dataloader_iterator._dataloader.dataset.fast_forward)
            for _ in range(self._iter):
                print("|")
                next(self.dataloader_iterator)
            self.dataloader_iterator._dataloader.dataset.fast_forward = False"""
        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()

            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and self._iter % self.val_interval == 0):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')
        return self.runner.model


LOOPS.register_module(name="IterBasedTrainLoop", module=IterBasedTrainLoopFastRestart, force=True)

