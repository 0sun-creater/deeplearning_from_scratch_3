class Variable:
    def __init__(self,data):
        if data is not None:    #추가
            if not isinstance(data, np.ndarray):
                raise TypeError("{}은 지원하지 않습니다.".format(type(data)))
        
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:   
            self.grad = np.ones_like(self.data)
        
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() 
            x, y = f.input, f.output 
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

class Function:
    def __call__(self, *inputs):  #별표
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) 	  #별표
        if not isinstance(ys, tuple): #튜플이 아닌 경우 지원
            ys = (ys, )
        outputs = [Variable[as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        
        #리스트의 원소가 하나라면 첫 번재 원소를 반환한다.
        return outputs if len(outputs) > 1 else outputs[0]
        
    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y



class Square(Function):
    def forward(self, x):
        return x**2


    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
    

#y=e^x 함수 구현
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def add(x0, x1):
    return Add()(x0,x1)



def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)


#수치 미분
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data - eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

#합성함수   
def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

def as_array(x):
    if np.isscallar(x):
        return np.array(x)
    return x

#실행코드
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))

a = A(x)
b = B(a)
c = C(b)

#계산 그래프의 노드들을 거꾸로 거슬러 올라간다.
assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x


'''
#역전파에 대응하는 Variable 클래스 구현
class Variable:
    def __init__(self,data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator #함수 가져오기
        if f is not None:
            x = f.input  #함수의 입력 가져오기
            x.grad = f.backward(self.grad) #함수의 backward 메서드 호출
            x.backward() #하나 앞 변수의 backward 메서드 호출 (재귀)




'''
