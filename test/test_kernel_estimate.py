import unittest
import torch
from torch.autograd import Variable
from estimators.kernel_density_estimate import KernelDensityEstimate


class TestKernelDensityEstimate(unittest.TestCase):

    def test_gaussian_kernel(self):
        """ Compare x=<1,1> with x_i = [i, j, -j]
        """
        kde = KernelDensityEstimate(delta=0)
        x_i = torch.FloatTensor([[1, 0], [0, 1], [0, -1]])
        x = torch.FloatTensor([1, 1])

        result = kde.gaussian_kernel(x, x_i)

        self.assertEqual(result[0, 0], 1)
        self.assertEqual(result[1, 0], result[0, 0])
        self.assertAlmostEqual(result[2, 0], 0.4472, places=4)

    def test_normalized_kernel(self):
        """ Compare x=<1,1> with x_i = [i, j, -j]
        """
        kde = KernelDensityEstimate(delta=0)
        x = torch.FloatTensor([1, 1])
        x_i = torch.FloatTensor([[1, 0], [0, 1], [0, -1]])

        distances = kde.gaussian_kernel(x, x_i)
        result = kde.normalized_kernel(distances)

        self.assertAlmostEqual(result[0, 0], 0.4086, places=4)
        self.assertAlmostEqual(result[1, 0], result[0, 0])
        self.assertAlmostEqual(result[2, 0], 0.1827, places=4)

    def test_normalized_kernel_variable(self):
        """ Compare x=<1,1> with x_i = [i, j, -j]
        """
        kde = KernelDensityEstimate(delta=0)
        x = Variable(torch.FloatTensor([1, 1]))
        x_i = Variable(torch.FloatTensor([[1, 0], [0, 1], [0, -1]]))

        distances = kde.gaussian_kernel(x, x_i)
        result = kde.normalized_kernel(distances)

        self.assertAlmostEqual(result.data[0, 0], 0.4086, places=4)
        self.assertAlmostEqual(result.data[1, 0], result.data[0, 0])
        self.assertAlmostEqual(result.data[2, 0], 0.1827, places=4)


if __name__ == "__main__":
    unittest.main()
