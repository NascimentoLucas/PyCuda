# imports necessários
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

# criando as matrizes
a = numpy.random.randn(4, 4)

# baixando a precisão do double
a = a.astype(numpy.float32)


# compila o kernel e já carrega para o device
# caso tenha algum erro no código será gerado um erro para o python
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

# é necessário achar a referência para o kernel criado
# dessa forma não é necessário copiar manualmente
# DtH ou HtD
func = mod.get_function("doublify")
func(cuda.InOut(a), block=(4, 4, 1))

print(str(a))


