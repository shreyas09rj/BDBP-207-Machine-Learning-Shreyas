#!/usr/bin/python


#implement of transpose of A

#
# import numpy as np
#
# # Define matrix A
# A = np.array([[1, 2, 3],
#               [4, 5, 6]])
#
#
# ATA =np.dot(A.T, A)
#
#
# print("\nA^T A:")
# print(ATA)



def transpose(matrix):
    rows = len(matrix)
    colums = len(matrix[0])

    T = []
    for j in range(colums):
        row = []
        for i in range(rows):
            row.append(matrix[i][j])
        T.append(row)
    return T


def matmul(A, B):
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            s = 0
            for k in range(len(B)):
                s += A[i][k] * B[k][j]
            row.append(s)
        result.append(row)
    return result



A = [
    [1, 2, 3],
    [4, 5, 6]
]

AT = transpose(A)
ATA = matmul(AT, A)

print("A^T * A =")
for row in ATA:
    print(row)








