class ConfusionMatrix(object):
    def __init__(self, matrix):
        self.count = len(matrix)
        self._matrix = []
        for mat in matrix:
            self._matrix.append([m for m in mat])
        self.normalize()

    def normalize(self):
        for i in range(self.count):
            sum=0
            for item in self[i]:
                sum+=item
            for j in range(self.count):
                    self._matrix[j][i]=(self._matrix[j][i]+0.0)/sum

    def print_matrix(self):
        for colum in self._matrix:
            for obj in colum:
                print obj,
            print "\n",

    def __getitem__(self, item):
        return [mat[int(item)] for mat in self._matrix]
