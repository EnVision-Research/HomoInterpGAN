from util.visualization import *

def test_layer_correlation():
    a=torch.rand((2,4,4))
    b=torch.rand((5,4,4))
    print(layer_correlation(a,b))

if __name__ == '__main__':
    test_layer_correlation()