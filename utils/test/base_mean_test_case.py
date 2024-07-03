from abc import ABC, abstractmethod

import torch


class BaseMeanTestCase(ABC):
    batch_shape = None

    @abstractmethod
    def create_mean(self, **kwargs): ...

    def test_forward_vec(self) -> None:
        test_x = torch.randn(4)
        mean = self.create_mean()
        if self.__class__.batch_shape is None:
            self.assertEqual(mean(test_x).shape, torch.Size([4]))
        else:
            self.assertEqual(
                mean(test_x).shape, torch.Size([*self.__class__.batch_shape, 4])
            )

    def test_forward_mat(self) -> None:
        test_x = torch.randn(4, 3)
        mean = self.create_mean()
        if self.__class__.batch_shape is None:
            self.assertEqual(mean(test_x).shape, torch.Size([4]))
        else:
            self.assertEqual(
                mean(test_x).shape, torch.Size([*self.__class__.batch_shape, 4])
            )

    def test_forward_mat_batch(self) -> None:
        test_x = torch.randn(3, 4, 3)
        mean = self.create_mean()
        if self.__class__.batch_shape is None:
            self.assertEqual(mean(test_x).shape, torch.Size([3, 4]))
        else:
            self.assertEqual(
                mean(test_x).shape, torch.Size([*self.__class__.batch_shape, 4])
            )

    def test_forward_mat_multi_batch(self) -> None:
        test_x = torch.randn(2, 3, 4, 3)
        mean = self.create_mean()
        self.assertEqual(mean(test_x).shape, torch.Size([2, 3, 4]))
