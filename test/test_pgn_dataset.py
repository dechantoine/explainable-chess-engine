import unittest

from torch import Tensor

from src.data.base_dataset import BoardItem
from src.data.pgn_dataset import PGNDataset

test_data_dir = "test/test_data"


class PGNDatasetTestCase(unittest.TestCase):
    def setUp(self):
        self.dataset = PGNDataset(
            root_dir=test_data_dir,
            include_draws=True,
            in_memory=False,
            winner=False,
            move_count=False,
        )

        self.dataset_in_memory = PGNDataset(
            root_dir=test_data_dir,
            include_draws=True,
            in_memory=True,
            num_workers=8,
            winner=False,
            move_count=False,
        )

        self.dataset_in_memory_return_moves = PGNDataset(
            root_dir=test_data_dir,
            include_draws=True,
            in_memory=True,
            num_workers=8,
            winner=False,
            move_count=True,
        )

        self.dataset_in_memory_return_outcome = PGNDataset(
            root_dir=test_data_dir,
            include_draws=True,
            in_memory=True,
            num_workers=8,
            winner=True,
            move_count=False,
        )

        self.dataset_in_memory_return_both = PGNDataset(
            root_dir=test_data_dir,
            include_draws=True,
            in_memory=True,
            num_workers=8,
            winner=True,
            move_count=True,
        )

        self.len_moves_najdorf = [73, 58, 44, 87]
        self.len_moves_tal = [60, 127, 82, 59, 45, 128]
        self.len_moves_morphy = [
            61,
            35,
            0,
            33,
            46,
            45,
            21,
            45,
            29,
            41,
            91,
            39,
            56,
            39,
            109,
            27,
            47,
            28,
        ]

        self.results_najdorf = [0, 0, -1, 0, 0]
        self.results_tal = [-1, 1, 0, 1, 1, 1]
        self.results_morphy = [1, 1, 0, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1]

        self.len_moves_najdorf_no_draws = [44]
        self.len_moves_tal_no_draws = [60, 127, 59, 45, 128]
        self.len_moves_morphy_no_draws = [
            61,
            35,
            0,
            33,
            46,
            45,
            21,
            45,
            29,
            41,
            91,
            39,
            56,
            39,
            109,
            27,
            47,
            28,
        ]

    def test_init(self):
        self.assertRaises(AttributeError, lambda: self.dataset.board_samples)
        self.assertRaises(AttributeError, lambda: self.dataset.moves_ids)
        self.assertRaises(AttributeError, lambda: self.dataset.total_moves)
        self.assertRaises(AttributeError, lambda: self.dataset.winners)

        self.assertIsNotNone(self.dataset_in_memory.board_samples)
        self.assertRaises(
            AttributeError, lambda: self.dataset_in_memory.moves_ids
        )
        self.assertRaises(
            AttributeError, lambda: self.dataset_in_memory.total_moves
        )
        self.assertRaises(AttributeError, lambda: self.dataset_in_memory.winners)

        self.assertIsNotNone(self.dataset_in_memory_return_moves.board_samples)
        self.assertIsNotNone(self.dataset_in_memory_return_moves.moves_ids)
        self.assertIsNotNone(self.dataset_in_memory_return_moves.total_moves)
        self.assertRaises(
            AttributeError, lambda: self.dataset_in_memory_return_moves.winners
        )

        self.assertIsNotNone(self.dataset_in_memory_return_outcome.board_samples)
        self.assertRaises(
            AttributeError,
            lambda: self.dataset_in_memory_return_outcome.moves_ids,
        )
        self.assertRaises(
            AttributeError,
            lambda: self.dataset_in_memory_return_outcome.total_moves,
        )
        self.assertIsNotNone(self.dataset_in_memory_return_outcome.winners)

        self.assertIsNotNone(self.dataset_in_memory_return_moves.board_samples)
        self.assertIsNotNone(self.dataset_in_memory_return_moves.moves_ids)
        self.assertIsNotNone(self.dataset_in_memory_return_moves.total_moves)
        self.assertIsNotNone(self.dataset_in_memory_return_outcome.winners)

    def test_get_boards_indices(self):
        for d in [
            self.dataset,
            self.dataset_in_memory,
            self.dataset_in_memory_return_moves,
            self.dataset_in_memory_return_outcome,
            self.dataset_in_memory_return_both,
        ]:
            indices = d.get_boards_indices(include_draws=True)
            self.assertIsInstance(indices, list)
            self.assertTrue(all(isinstance(x, tuple) for x in indices))
            self.assertTrue(all(len(x) == 3 for x in indices))
            self.assertTrue(all(isinstance(x[i], int) for x in indices for i in range(3)))

            self.assertTrue(len([x for x in indices if x[0] == 0]) == sum(self.len_moves_morphy))
            self.assertTrue(len([x for x in indices if x[0] == 1]) == sum(self.len_moves_najdorf))
            self.assertTrue(len([x for x in indices if x[0] == 2]) == sum(self.len_moves_tal))

            self.assertTrue(all(
                len([x for x in indices if x[0] == 0 and x[1] == i])
                == self.len_moves_morphy[i]
                for i in range(len(self.len_moves_morphy))
            ))
            self.assertTrue(all(
                len([x for x in indices if x[0] == 1 and x[1] == i])
                == self.len_moves_najdorf[i]
                for i in range(len(self.len_moves_najdorf))
            ))
            self.assertTrue(all(
                len([x for x in indices if x[0] == 2 and x[1] == i])
                == self.len_moves_tal[i]
                for i in range(len(self.len_moves_tal))
            ))

            indices_no_draws = d.get_boards_indices(
                include_draws=False
            )
            self.assertTrue(len([x for x in indices_no_draws if x[0] == 0]) == sum(
                self.len_moves_morphy_no_draws
            ))
            self.assertTrue(len([x for x in indices_no_draws if x[0] == 1]) == sum(
                self.len_moves_najdorf_no_draws
            ))
            self.assertTrue(len([x for x in indices_no_draws if x[0] == 2]) == sum(
                self.len_moves_tal_no_draws
            ))

    def test_retrieve_board(self):
        for d in [
            self.dataset,
            self.dataset_in_memory,
            self.dataset_in_memory_return_moves,
            self.dataset_in_memory_return_outcome,
            self.dataset_in_memory_return_both,
        ]:
            board, move_id, total_moves, result = d.retrieve_board(2)
            self.assertIsInstance(move_id, int)
            self.assertIsInstance(total_moves, int)
            self.assertIsInstance(result, int)

    def test_getitem(self):
        board_item = self.dataset[0]
        self.assertIsInstance(board_item, BoardItem)
        self.assertIsInstance(board_item.board, Tensor)
        self.assertIsInstance(board_item.active_color, Tensor)
        self.assertIsInstance(board_item.castling, Tensor)

        board_item = self.dataset[Tensor([0])]
        self.assertIsInstance(board_item, BoardItem)
        self.assertIsInstance(board_item.board, Tensor)
        self.assertIsInstance(board_item.active_color, Tensor)
        self.assertIsInstance(board_item.castling, Tensor)

        self.dataset.move_count = True
        board_item = self.dataset[0]
        self.assertIsInstance(board_item, BoardItem)
        self.assertIsInstance(board_item.board, Tensor)
        self.assertIsInstance(board_item.active_color, Tensor)
        self.assertIsInstance(board_item.castling, Tensor)
        self.assertIsInstance(board_item.move_id, Tensor)
        self.assertIsInstance(board_item.total_moves, Tensor)

        self.dataset.winner = True
        board_item = self.dataset[0]
        self.assertIsInstance(board_item, BoardItem)
        self.assertIsInstance(board_item.board, Tensor)
        self.assertIsInstance(board_item.active_color, Tensor)
        self.assertIsInstance(board_item.castling, Tensor)
        self.assertIsInstance(board_item.move_id, Tensor)
        self.assertIsInstance(board_item.total_moves, Tensor)
        self.assertIsInstance(board_item.winner, Tensor)

        self.assertEqual(board_item.board.shape, (12, 8, 8))
        self.assertEqual(board_item.active_color.shape, (1,))
        self.assertEqual(board_item.castling.shape, (4,))
        self.assertEqual(board_item.move_id.shape, (1,))
        self.assertEqual(board_item.total_moves.shape, (1,))
        self.assertEqual(board_item.winner.shape, (1,))

        board_item = self.dataset_in_memory_return_both[0]
        self.assertIsInstance(board_item, BoardItem)
        self.assertIsInstance(board_item.board, Tensor)
        self.assertIsInstance(board_item.active_color, Tensor)
        self.assertIsInstance(board_item.castling, Tensor)
        self.assertIsInstance(board_item.move_id, Tensor)
        self.assertIsInstance(board_item.total_moves, Tensor)
        self.assertIsInstance(board_item.winner, Tensor)

        self.assertEqual(board_item.board.shape, (12, 8, 8))
        self.assertEqual(board_item.active_color.shape, (1,))
        self.assertEqual(board_item.castling.shape, (4,))
        self.assertEqual(board_item.move_id.shape, (1,))
        self.assertEqual(board_item.total_moves.shape, (1,))
        self.assertEqual(board_item.winner.shape, (1,))

        board_item = self.dataset_in_memory_return_moves[0]
        self.assertIsInstance(board_item, BoardItem)
        self.assertIsInstance(board_item.board, Tensor)
        self.assertIsInstance(board_item.active_color, Tensor)
        self.assertIsInstance(board_item.castling, Tensor)
        self.assertIsInstance(board_item.move_id, Tensor)
        self.assertIsInstance(board_item.total_moves, Tensor)

        self.assertEqual(board_item.board.shape, (12, 8, 8))
        self.assertEqual(board_item.active_color.shape, (1,))
        self.assertEqual(board_item.castling.shape, (4,))
        self.assertEqual(board_item.move_id.shape, (1,))
        self.assertEqual(board_item.total_moves.shape, (1,))

        board_item = self.dataset_in_memory_return_outcome[0]
        self.assertIsInstance(board_item, BoardItem)
        self.assertIsInstance(board_item.board, Tensor)
        self.assertIsInstance(board_item.active_color, Tensor)
        self.assertIsInstance(board_item.castling, Tensor)
        self.assertIsInstance(board_item.winner, Tensor)

        self.assertEqual(board_item.board.shape, (12, 8, 8))
        self.assertEqual(board_item.active_color.shape, (1,))
        self.assertEqual(board_item.castling.shape, (4,))
        self.assertEqual(board_item.winner.shape, (1,))

    def test_getitems(self):
        boards_item = self.dataset.__getitems__([0, 1, 2])
        self.assertIsInstance(boards_item, BoardItem)
        self.assertIsInstance(boards_item.board, Tensor)
        self.assertIsInstance(boards_item.active_color, Tensor)
        self.assertIsInstance(boards_item.castling, Tensor)

        boards_item = self.dataset.__getitems__(Tensor([0, 1, 2]))
        self.assertIsInstance(boards_item, BoardItem)
        self.assertIsInstance(boards_item.board, Tensor)
        self.assertIsInstance(boards_item.active_color, Tensor)
        self.assertIsInstance(boards_item.castling, Tensor)

        self.dataset.move_count = True
        boards_item = self.dataset.__getitems__([0, 1, 2])
        self.assertIsInstance(boards_item, BoardItem)
        self.assertIsInstance(boards_item.board, Tensor)
        self.assertIsInstance(boards_item.active_color, Tensor)
        self.assertIsInstance(boards_item.castling, Tensor)
        self.assertIsInstance(boards_item.move_id, Tensor)
        self.assertIsInstance(boards_item.total_moves, Tensor)

        self.dataset.winner = True
        boards_item = self.dataset.__getitems__([0, 1, 2])
        self.assertIsInstance(boards_item, BoardItem)
        self.assertIsInstance(boards_item.board, Tensor)
        self.assertIsInstance(boards_item.active_color, Tensor)
        self.assertIsInstance(boards_item.castling, Tensor)
        self.assertIsInstance(boards_item.move_id, Tensor)
        self.assertIsInstance(boards_item.total_moves, Tensor)
        self.assertIsInstance(boards_item.winner, Tensor)

        self.assertEqual(boards_item.board.shape, (3, 12, 8, 8))
        self.assertEqual(boards_item.active_color.shape, (3, 1))
        self.assertEqual(boards_item.castling.shape, (3, 4))
        self.assertEqual(boards_item.move_id.shape, (3, 1))
        self.assertEqual(boards_item.total_moves.shape, (3, 1))
        self.assertEqual(boards_item.winner.shape, (3, 1))

        boards_item = self.dataset_in_memory.__getitems__([0, 1, 2])
        self.assertIsInstance(boards_item, BoardItem)
        self.assertIsInstance(boards_item.board, Tensor)
        self.assertIsInstance(boards_item.active_color, Tensor)
        self.assertIsInstance(boards_item.castling, Tensor)

        self.assertEqual(boards_item.board.shape, (3, 12, 8, 8))
        self.assertEqual(boards_item.active_color.shape, (3, 1))
        self.assertEqual(boards_item.castling.shape, (3, 4))

        boards_item = self.dataset_in_memory_return_moves.__getitems__([0, 1, 2])
        self.assertIsInstance(boards_item, BoardItem)
        self.assertIsInstance(boards_item.board, Tensor)
        self.assertIsInstance(boards_item.active_color, Tensor)
        self.assertIsInstance(boards_item.castling, Tensor)
        self.assertIsInstance(boards_item.move_id, Tensor)
        self.assertIsInstance(boards_item.total_moves, Tensor)

        self.assertEqual(boards_item.board.shape, (3, 12, 8, 8))
        self.assertEqual(boards_item.active_color.shape, (3, 1))
        self.assertEqual(boards_item.castling.shape, (3, 4))
        self.assertEqual(boards_item.move_id.shape, (3, 1))
        self.assertEqual(boards_item.total_moves.shape, (3, 1))

        boards_item = self.dataset_in_memory_return_outcome.__getitems__([0, 1, 2])
        self.assertIsInstance(boards_item, BoardItem)
        self.assertIsInstance(boards_item.board, Tensor)
        self.assertIsInstance(boards_item.active_color, Tensor)
        self.assertIsInstance(boards_item.castling, Tensor)
        self.assertIsInstance(boards_item.winner, Tensor)

        self.assertEqual(boards_item.board.shape, (3, 12, 8, 8))
        self.assertEqual(boards_item.active_color.shape, (3, 1))
        self.assertEqual(boards_item.castling.shape, (3, 4))
        self.assertEqual(boards_item.winner.shape, (3, 1))

        boards_item = self.dataset_in_memory_return_both.__getitems__([0, 1, 2])
        self.assertIsInstance(boards_item, BoardItem)
        self.assertIsInstance(boards_item.board, Tensor)
        self.assertIsInstance(boards_item.active_color, Tensor)
        self.assertIsInstance(boards_item.castling, Tensor)
        self.assertIsInstance(boards_item.move_id, Tensor)
        self.assertIsInstance(boards_item.total_moves, Tensor)
        self.assertIsInstance(boards_item.winner, Tensor)

        self.assertEqual(boards_item.board.shape, (3, 12, 8, 8))
        self.assertEqual(boards_item.active_color.shape, (3, 1))
        self.assertEqual(boards_item.castling.shape, (3, 4))
        self.assertEqual(boards_item.move_id.shape, (3, 1))
        self.assertEqual(boards_item.total_moves.shape, (3, 1))
        self.assertEqual(boards_item.winner.shape, (3, 1))
