import torch


def main():
    print("HI")
    t = torch.Tensor([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
        [[28, 29, 30], [31, 32, 33], [34, 35, 56]],
    ])
    print(t[0][0][0])
    t[:, 0][0] = 999
    print(t[0][0][0])
    print(t[:, 0])

    # print(t.shape)
    # print(type(t.shape))
    # print(t.shape == (2, 3))
    # print(t.shape == Size([2, 3]))
    # print(Size([1, 2, 3]))


if __name__ == '__main__':
    main()
    # test()
