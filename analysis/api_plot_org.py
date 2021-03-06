"""Plotting functions imported from bvmpc module."""

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class GridFig:
    """Handles gridded figures."""

    def __init__(
            self, rows, cols=4,
            size_multiplier=5, wspace=0.3, hspace=0.3,
            tight_layout=False):
        """
        Set up the grid specifications.

        size_multiplier, wspace, and hspace are used for spacing

        """
        self.fig = plt.figure(
            figsize=(cols * size_multiplier,
                     rows * size_multiplier),
            tight_layout=tight_layout)
        self.gs = gridspec.GridSpec(rows, cols, wspace=wspace, hspace=hspace)
        self.idx = 0
        self.rows = rows
        self.cols = cols

    def get_ax(self, row_idx, col_idx):
        """Add subplot with standard 1x1 gs -> returns ax."""
        ax = self.fig.add_subplot(self.gs[row_idx, col_idx])
        return ax

    def get_multi_ax(self, row_start, row_end, col_start, col_end):
        """Add subplot with custom gs sizes -> returns ax."""
        ax = self.fig.add_subplot(
            self.gs[row_start:row_end, col_start:col_end])
        plt.subplots_adjust(top=0.85)
        return ax

    def save_fig(self, name, out_dir):
        """
        Names and saves figure.

        """
        
        out_name = str(name) + ".png"
        print("Saved figure to {}".format(
            os.path.join(out_dir, out_name)))
        self.fig.savefig(os.path.join(out_dir, out_name), dpi=400)
        plt.close()
        return

    def get_fig(self):
        """Return the figure object in this class."""
        return self.fig

    def get_next(self, along_rows=True):
        """
        Get next index along rows or columns.

        along rows:
        1   2   3   4   5   6
        7   8   9   10  11  12  ...

        else:
        1   3   5   7   9   11
        2   4   6   8   10  12  ...

        """
        if along_rows:
            row_idx = (self.idx // self.cols)
            col_idx = (self.idx % self.cols)

            print(row_idx, col_idx)
            ax = self.get_ax(row_idx, col_idx)
        else:
            row_idx = (self.idx % self.rows)
            col_idx = (self.idx // self.rows)

            print(row_idx, col_idx)
            ax = self.get_ax(row_idx, col_idx)
        self._increment()
        return ax

    def get_next_snake(self):
        """
        Get the next index in a snake like pattern.

        1   2   5   6   9   10  ...
        3   4   7   8   11  12  ..

        """
        if self.rows != 2:
            print("ERROR: Can't get snake unless there are two rows")
        i = self.idx
        row_idx = (i // 2) % 2
        col_idx = (i // 2) + (i % 2) - row_idx
        ax = self.get_ax(row_idx, col_idx)
        self._increment()
        return ax

    def _increment(self):
        """Private function to increase the internal idx counter."""
        self.idx = self.idx + 1
        if self.idx == self.rows * self.cols:
            print("Looping")
            self.idx = 0
