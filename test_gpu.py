import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    x = torch.randn(3, 3).to(device)
    print("Tensor on device:", x)

    if torch.cuda.is_available() and x.is_cuda:
        print("GPU is working!")
    else:
        print("GPU not used, check your environment.")

if __name__ == "__main__":
    main()