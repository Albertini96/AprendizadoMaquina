#Get the transpose of matrix
def transposeMatrix(mat):
    #Get number of columns
    c = len(mat[0])
   
    #Get number of rows
    r = len(mat)
    #Transposing Matrix
    return [[mat[i][j] for i in range(r)] for j in range(c)]

#Get the cofactor matrix
def getCofactorMatrix(mat, i, j):
    #Get desired rows
    rows = mat[:i]+mat[i+1:]
    #Get desired columns of rows
    minor = [row[:j] + row[j+1:] for row in rows]
    return minor

#Calculate the determinant of matrix
def getDeternminant(mat):
    
    det = 0
    
    #If matrix is 2x2 then return determinant
    if len(mat) == 2:
        return (mat[0][0]*mat[1][1])-(mat[0][1]*mat[1][0])

    for c in range(len(mat)):
        cal = ((-1)**c)*mat[0][c]
        g_det = getDeternminant(getCofactorMatrix(mat,0,c))
        det += cal * g_det
        
    return det

def getMatrixInverse(mat):
    det = getDeternminant(mat)

    #If matrix is 2x2 then return inverse
    if len(mat) == 2:
        return [[mat[1][1]/det, -1*mat[0][1]/det], [-1*mat[1][0]/det, mat[0][0]/det]]

    cofactors = []

    #Calculating cofactors
    for r in range(len(mat)):
        cofactorRow = []
        for c in range(len(mat)):
            minor = getCofactorMatrix(mat, r, c)
            
            cofactorRow.append(((-1)**(r+c)) * getDeternminant(minor))
            
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    
    #Calculating inverted matrix
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/det
            
    return cofactors

def multMatrix(mat_b, mat_a):
    
    rows_a = len(mat_a)
    cols_a = len(mat_a[0])
    cols_b = len(mat_b[0])

    ret = [[0 for row in range(cols_b)] for col in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                ret[i][j] += mat_a[i][k] * mat_b[k][j]
                
    return ret