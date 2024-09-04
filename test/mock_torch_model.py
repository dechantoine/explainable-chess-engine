import torch


class MockModel(torch.nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(in_features=12 * 8 * 8 + 1 + 4, out_features=1)

    def forward(self, x):
        board, color, castling = x
        board = board.float()
        color = color.float()

        castling = castling.float()
        board = self.flatten(board)

        x = torch.cat((board, color, castling), dim=1)
        return self.linear(x)

    def model_hash(self):
        return "mock_model"
