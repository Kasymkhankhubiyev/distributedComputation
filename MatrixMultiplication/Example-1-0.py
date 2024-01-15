from numpy import empty

# Считываем из файла число строк M и число столбцов N
f1 = open('in.dat', 'r')
N = int(f1.readline())
M = int(f1.readline())
f1.close()

# Выделяем под матрицы A,x и b соответствующее место в памяти
A = empty((M,N)); x = empty(N); b = empty(M)
	
# Считываем из файла матрицу A
f2 = open('AData.dat', 'r')
for j in range(M) :
    for i in range(N) :
        A[j,i] = float(f2.readline())
f2.close()
		
# Считываем из файла вектор x
f3 = open('xData.dat', 'r')
for i in range(N) :
    x[i] = float(f3.readline())
f3.close()
	
# Основная вычислительная часть программы
# Умножаем матрцу A на вектор x
# ------------------------------------------------------------
for j in range(M) :
    b[j] = 0.
    for i in range(N) :
        b[j] = b[j] + A[j,i]*x[i]
#------------------------------------------------------------
	
# Сохраняем результат вычислений в файл
f4 = open('Results.dat', 'w')
for j in range(M) :
    f4.write(str(b[j])+'\n')
f4.close()

print(b)