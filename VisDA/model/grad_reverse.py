from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx,x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx,grad_output):
        return (grad_output * -1)


def grad_reverse():
    return GradReverse.apply