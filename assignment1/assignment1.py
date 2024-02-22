import sys
import os
import math
import numpy as np

#function to extract mean vector/array from a matrix
def meanVector(matrix):
    meanVector = []

    for row in matrix:
        N = len(row)
        mean = (sum(row)/N)
        meanVector.append(mean)

    return meanVector
        
#function to calulate difference matrix
def differenceMatrix(inputMatrix, meanVector):
    differenceMatrix = []

    for inputRow in inputMatrix:
        outputRow = []

        for currentColumn in inputRow:
            meanValue = meanVector[inputMatrix.index(inputRow)]
            outputRow.append(currentColumn - meanValue)

        differenceMatrix.append(outputRow)

    return differenceMatrix

#matrix multiplication
def matrix_multiplication(matrix1, matrix2):
    result = []

    # Iterate through rows of the first matrix
    for i in range(len(matrix1)):
        row = []

        # Iterate through columns of the second matrix
        for j in range(len(matrix2[0])):
            sum = 0

            # Iterate through rows of the second matrix
            for k in range(len(matrix2)):
                sum += matrix1[i][k] * matrix2[k][j]

            row.append(sum)
        result.append(row)

    return result

#function to transpose a matrix
def transposeMatrix(matrix):
    #get the dimensions of the original matrix
    rows = len(matrix)
    columns = len(matrix[0])

    #create an empty transposed matrix
    transposed_matrix = [[0 for _ in range(rows)] for _ in range(columns)]

    #fill the transposed matrix with the values from the original matrix
    for i in range(rows):
        for j in range(columns):
            transposed_matrix[j][i] = matrix[i][j]

    return transposed_matrix

#function to calulate APPROXIMATED covariance matrix C = DD^T
def covarianceMatrix(differenceMatrix):
    
    transposedMatrix = transposeMatrix(differenceMatrix)
    
    covarianceMatrix = matrix_multiplication(differenceMatrix, transposedMatrix)
    
    return covarianceMatrix

def left_shift_matrix(matrix, shift):
    shifted_matrix = []
    for row in matrix:
        n = len(row)
        shifted_row = [0] * n
        for i in range(n):
            shifted_row[i] = row[(i + shift) % n]
        shifted_matrix.append(shifted_row)
    return shifted_matrix

def left_shift_array(array, shift):
    n = len(array)
    shifted_array = [0] * n
    for i in range(n):
        shifted_array[i] = array[(i + shift) % n]
    return shifted_array

def projection(eigenValues, eigenVectors, originalMatrix):
    precision = int(input("Input desired decimal precision\n"))
    originalPrecision = len(str(originalMatrix[0][0]))
    if precision > originalPrecision:
        precision = originalPrecision
    k = len(eigenValues)
    result = []
    
    for i in range(k):
        #convert array/row of the selected matrix into matrix form for multiplication
        tempEigenValueMatrix = []
        tempEigenValueMatrix.append([eigenValues[i]])

        tempEigenVectorsMatrix = []
        tempEigenVectorsMatrix.append(eigenVectors[i])

        #pk*ek eigen value times eigen vector
        pkTimeEk = matrix_multiplication(tempEigenValueMatrix, tempEigenVectorsMatrix)

        #pk*ek*vi the product of the above with the original data matrix
        realNumbers = []
        for value in matrix_multiplication(pkTimeEk, originalMatrix)[0]:
            realNumbers.append(round(value.real, precision))

        result.append(realNumbers)
            

    finalResult = transposeMatrix(result)
    with open("pca_result.txt", 'w') as file:
        for row in finalResult:
            cleanText = str(row).replace("[", "").replace("]", "").replace(",", "")
            file.write(cleanText + "\n")


def PCA(matrix):

    meanVectorVbar = meanVector(matrix)
    diffMatrix = differenceMatrix(matrix, meanVectorVbar)
    covMatrix = covarianceMatrix(diffMatrix)
    eigenValues, eigenVectors = np.linalg.eig(covMatrix)

    sortedEigenValues = sorted(eigenValues)

    #spits eigenvalues into a file and asks the user to determine their truncation
    with open("sortedEigenValues.txt", 'w') as file:   
        for value in sortedEigenValues:
                file.write(str(value) + '\n')
    
    truncationPoint = int(input("Please input the desired truncation point of the eigenvalues.\nLook a the temporary sortedEigenValues.txt file in an editor with a row counter a type a cut off index. \n(ordered from least to greatest)(selected index will be included in deletion))\n"))
    #deletes the temporary file
    file.close()
    os.remove("sortedEigenValues.txt")

    for i in range(truncationPoint):
        deletionValue = sortedEigenValues[i]
        index = np.where(eigenValues == deletionValue)[0]

        eigenVectors = np.delete(eigenVectors, index, axis=0)
        eigenValues = np.delete(eigenValues, index)
    
    #shiftedEigenVectors = left_shift_matrix(eigenVectors, eigenVectors.argmax())
    #shiftedEigenValues = left_shift_array(eigenValues, eigenVectors.argmax())

    projection(eigenValues, eigenVectors, matrix)

    
def DCT(matrix):

    umatrix = []
    for row in matrix:
        n = len(row)
        alpha0 = math.sqrt(1/n)
        alphai = math.sqrt(2/n)
        
        urow = []
        for i in range(n):
            sum = 0
            u = 0
            for j in range(n):
                sum += row[j]*math.cos((((2*j)+1)*i*math.pi)/(2*n))

            if(i == 0):
                u = alpha0 * sum
                urow.append(u)
            else:
                u = alphai * sum
                urow.append(u)
        
        umatrix.append(urow)
    sortedMatrix = []
    for row in umatrix:
        sortedValues = sorted(row)
        sortedMatrix.append(sortedValues)
    
    partitionVector = meanVector(sortedMatrix)

    k_values = []
    for index in range(len(sortedMatrix)):
        deletionIndex = np.searchsorted(sortedMatrix[index], partitionVector[index])
        k_values.append(deletionIndex)
    print(k_values)
    truncationPoint = int(input("k_values are printed, select a truncation point (int) \n"))


    k_selected_row = sortedMatrix[k_values.index(truncationPoint)]

    for i in range(truncationPoint):
        deletionValue = k_selected_row[i]
        index = umatrix[k_values.index(truncationPoint)].index(deletionValue)

        for row in range(len(umatrix)):
            umatrix[row].pop(index)

    precision = len(str(matrix[0][0]))
    result = []
    for row in umatrix:
        realNumbers = []
        for index in row:
            realNumbers.append(round(index.real, precision))
        result.append(realNumbers)

    with open("dct_result.txt", 'w') as file:
        for row in result:
            cleanText = str(row).replace("[", "").replace("]", "").replace(",", "")
            file.write(cleanText + "\n")
    

def main():
    inputMatrix = []

    #parse file into a matrix
    with open(sys.argv[1], 'r') as file:
        for column in file:
            row = list(map(float, column.split()))
            inputMatrix.append(row)

    #originally based on the lecture example with row as attributes and column as N
    #so we must transpose the input matrix into that form then retranspose it back into proper form
    #after the reduction is completed
    #PCA(transposeMatrix(inputMatrix))

    DCT(inputMatrix)
  

if __name__ == "__main__":
    main()