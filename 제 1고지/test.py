import unittest
import deep

def numerical_diff(f, x, eps=1e-4):
    x0 = deep.Variable(x.data - eps)
    x1 = deep.Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = deep.Variable(np.array(2.0))
        y = deep.square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = deep.Variable(np.array(3.0))
        y = deep.square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = deep.Variable(np.random.rand(1))
        y = deep.square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)

    
