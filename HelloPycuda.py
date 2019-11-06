# imports necessários
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

# criando as matrizes
a = numpy.random.randn(4, 4)

# baixando a precisão do double
a = a.astype(numpy.float32)


# alocando memória no device
a_gpu = cuda.mem_alloc(a.nbytes)

# copiando para o device
cuda.memcpy_htod(a_gpu, a)

# compila o kernel e já carrega para o device
# caso tenha algum erro no código será gerado um erro
mod = SourceModule("""
//Dentro do kernel essa linha já conta como a terceira
//Por exemplo se retirar o 2, o erro será na linha 6
__device__ float doubleValue(float value){
    return value * 2;// tire o 2 e veja que o erro será na linha 6
} 
    
__global__ void doublify(float *a)
{
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] = doubleValue(idx);    
    printf("\\né possível usar prints %.1u ", idx);
}
  """)

# é necessário achar a referência para o kernel criado acima
# isso é feito com a linha abaixo
func = mod.get_function("doublify")

# a chamada para o kernel ser executado é feito abaixo,
# também é declarado o número de bloco e threads por bloco
func(a_gpu, block=(4, 4, 1))

a_doubled = numpy.empty_like(a)
# copiando de volta a matriz
cuda.memcpy_dtoh(a_doubled, a_gpu)

print(a_doubled)


